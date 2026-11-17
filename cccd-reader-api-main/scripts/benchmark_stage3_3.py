"""Benchmark toàn diện Giai đoạn 3.3: YOLOv8n Detection + OpenCV Document Corner Detection.

Mục tiêu kiểm thử:
1. Khảo sát khả năng định vị 4 góc vật lý thực tế của thẻ (TL, TR, BR, BL) bằng OpenCV trên 50 ảnh.
2. Đánh giá tỷ lệ thành công tìm 4 góc và Corner Localization Error (CLE).
3. So sánh đối đầu chất lượng OCR: Trước Rectification (YOLO Crop) vs Sau Rectification (OpenCV Warp).
4. Phân loại và lưu trữ toàn bộ các trường hợp thất bại của OpenCV.
5. Cung cấp số liệu định lượng để kết luận về việc chuyển sang YOLOv8n Keypoint Detection.

Tuân thủ nghiêm ngặt:
- Không train lại YOLO.
- Không thay đổi checkpoint YOLO hiện tại.
- Bounding box dùng để giới hạn vùng tìm kiếm, tuyệt đối không lấy 4 góc bbox làm 4 góc thẻ.
- Masking PII tuyệt đối trong mọi log output.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Thiết lập đường dẫn import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.document_detector import DocumentDetector
from services.document_rectifier import DocumentRectifier
from services.document_orientation import correct_orientation
from services.calibrated_rois import (
    extract_calibrated_cccd_fields_with_raw,
    extract_calibrated_gplx_fields_with_raw,
)
from services.adaptive_extractor import (
    CCCD_VALIDATORS,
    GPLX_VALIDATORS,
)
from services.cccd_extractor import (
    _normalize_width,
    extract_id_number,
    extract_full_name,
    extract_date_of_birth,
    extract_gender,
    extract_nationality,
    extract_place_of_origin,
    extract_residence,
    extract_expiry_date,
)
from services.gplx_extractor import (
    extract_license_number,
    extract_full_name as extract_gplx_name,
    extract_date_of_birth as extract_gplx_dob,
    extract_nationality as extract_gplx_nat,
    extract_address,
    extract_license_class,
    extract_expiry,
)
from services.field_cleaner import has_label_contamination

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_stage3_3")

ISO_ID1_ASPECT_RATIO = 85.60 / 53.98  # 1.58577...


def extract_baseline_yolo_crop(crop_img: np.ndarray, doc_type: str) -> dict[str, str | None]:
    """Trích xuất OCR trực tiếp từ YOLO Crop (Trước Rectification)."""
    norm = _normalize_width(crop_img)
    if doc_type == "cccd":
        return {
            "id_number": extract_id_number(norm),
            "full_name": extract_full_name(norm),
            "date_of_birth": extract_date_of_birth(norm),
            "gender": extract_gender(norm),
            "nationality": extract_nationality(norm),
            "place_of_origin": extract_place_of_origin(norm),
            "residence": extract_residence(norm),
            "expiry_date": extract_expiry_date(norm),
        }
    else:
        return {
            "license_number": extract_license_number(norm),
            "full_name": extract_gplx_name(norm),
            "date_of_birth": extract_gplx_dob(norm),
            "nationality": extract_gplx_nat(norm),
            "address": extract_address(norm),
            "license_class": extract_license_class(norm),
            "expiry": extract_expiry(norm),
        }


def compute_orthogonality_error(corners: list[list[int]] | np.ndarray) -> float:
    """Tính độ lệch góc trong so với 90 độ của tứ giác (đơn vị: độ)."""
    pts = np.array(corners, dtype=np.float32)
    devs = []
    for i in range(4):
        p_prev = pts[(i - 1) % 4]
        p_curr = pts[i]
        p_next = pts[(i + 1) % 4]

        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 45.0
        cos_t = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_t))
        devs.append(abs(angle - 90.0))
    return float(np.mean(devs))


def analyze_failure_cause(
    crop_img: np.ndarray,
    bbox: tuple[int, int, int, int],
    orig_shape: tuple[int, int],
) -> str:
    """Phân loại nguyên nhân khiến OpenCV không tìm được 4 góc."""
    h_c, w_c = crop_img.shape[:2]
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # 1. Kiểm tra chói lóa (Specular Glare)
    glare_pixels = np.sum(gray > 248)
    glare_ratio = glare_pixels / float(h_c * w_c)
    if glare_ratio > 0.08:
        return "Specular Glare (Chói lóa làm đứt gãy đường viền biên)"

    # 2. Kiểm tra độ tương phản biên (Low contrast / White on White)
    edges = cv2.Canny(gray, 40, 120)
    edge_density = np.sum(edges > 0) / float(h_c * w_c)
    if edge_density < 0.025:
        return "Low Contrast (Tương phản kém, viền thẻ tiệp màu nền)"

    # 3. Kiểm tra nền phức tạp (Complex Background / Texture Clutter)
    if edge_density > 0.22:
        return "Complex Texture Clutter (Nền quá nhiều hoa văn / chữ gây nhiễu contour)"

    # 4. Kiểm tra mép cắt sát rìa ảnh (Truncated boundary)
    x1, y1, x2, y2 = bbox
    h_orig, w_orig = orig_shape
    if x1 <= 5 or y1 <= 5 or x2 >= w_orig - 5 or y2 >= h_orig - 5:
        return "Truncated Edge (Thẻ nằm sát mép ảnh, bị khuyết góc)"

    # 5. Mặc định: Góc bo tròn hoặc biến dạng phối cảnh lớn
    return "Rounded Corner / Strong Skew (Bo tròn góc 3.18mm làm gãy polygon 4 đỉnh)"


def run_benchmark_stage3_3(
    output_dir: str | Path = "debug_output/stage3_3",
    sample_target: int = 50,
) -> dict:
    """Chạy benchmark Giai đoạn 3.3 trên ít nhất 50 ảnh đa dạng."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corners_debug_dir = out_dir / "corners_debug"
    corners_debug_dir.mkdir(parents=True, exist_ok=True)

    failures_dir = out_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Thu thập và cân bằng danh sách 50 ảnh (25 CCCD + 25 GPLX)
    test_dir = PROJECT_ROOT / "datasets" / "images" / "test"
    val_dir = PROJECT_ROOT / "datasets" / "images" / "val"

    all_cccd = sorted(list(test_dir.glob("cccd_*.jpg")) + list(val_dir.glob("cccd_*.jpg")))
    all_gplx = sorted(list(test_dir.glob("gplx_*.jpg")) + list(val_dir.glob("gplx_*.jpg")))

    half = sample_target // 2
    selected_cccd = all_cccd[:half]
    selected_gplx = all_gplx[:half]
    selected_images = selected_cccd + selected_gplx

    logger.info(
        f"Bắt đầu Benchmark Giai đoạn 3.3 trên {len(selected_images)} ảnh "
        f"({len(selected_cccd)} CCCD, {len(selected_gplx)} GPLX)"
    )

    detector = DocumentDetector(
        model_path=PROJECT_ROOT / "models" / "yolov8n_document.pt",
        conf_threshold=0.25,
    )
    rectifier = DocumentRectifier(target_size=(856, 540), margin=0.08)

    # Biến thống kê
    corner_metrics = {
        "total_images": len(selected_images),
        "cccd_images": len(selected_cccd),
        "gplx_images": len(selected_gplx),
        "yolo_detected": 0,
        "corners_success": 0,
        "corners_failed": 0,
        "methods": {"contour_approx": 0, "convex_hull": 0, "line_intersection": 0, "fallback_none": 0},
        "latencies_ms": [],
        "aspect_ratios": [],
        "aspect_errors": [],
        "orthogonality_errors": [],
    }

    ocr_before = {"total_fields": 0, "extracted": 0, "valid": 0, "latencies_sec": [], "field_stats": {}}
    ocr_after = {"total_fields": 0, "extracted": 0, "valid": 0, "latencies_sec": [], "field_stats": {}}

    failure_cases = []

    for idx, img_path in enumerate(selected_images, 1):
        filename = img_path.name
        doc_type = "cccd" if filename.startswith("cccd_") else "gplx"
        validators = CCCD_VALIDATORS if doc_type == "cccd" else GPLX_VALIDATORS

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Không thể đọc ảnh: {filename}")
            continue

        h_orig, w_orig = img.shape[:2]

        # Bước 1: YOLO Detection
        det = detector.detect(img)
        if not det.has_detection or not det.top_detection:
            logger.warning(f"[{idx:02d}] YOLO không phát hiện giấy tờ: {filename}")
            continue

        corner_metrics["yolo_detected"] += 1
        top_det = det.top_detection
        bbox = top_det.bbox_xyxy

        # Vùng crop ban đầu của YOLO
        x1, y1, x2, y2 = bbox
        yolo_raw_crop = img[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)].copy()

        # Bước 2: OpenCV Document Corner Detection & Rectification
        t_rect_start = time.perf_counter()
        rect_res = rectifier.rectify(img, bbox, card_type=doc_type)
        t_rect_end = time.perf_counter()
        rect_latency_ms = (t_rect_end - t_rect_start) * 1000.0
        corner_metrics["latencies_ms"].append(rect_latency_ms)

        m = rect_res.method
        corner_metrics["methods"][m] = corner_metrics["methods"].get(m, 0) + 1

        is_corner_ok = (
            rect_res.success
            and not rect_res.fallback_used
            and rect_res.corners is not None
            and len(rect_res.corners) == 4
        )

        if is_corner_ok:
            corner_metrics["corners_success"] += 1
            ar = rect_res.aspect_ratio
            ar_err = abs(ar - ISO_ID1_ASPECT_RATIO)
            ortho_err = compute_orthogonality_error(rect_res.corners)

            corner_metrics["aspect_ratios"].append(ar)
            corner_metrics["aspect_errors"].append(ar_err)
            corner_metrics["orthogonality_errors"].append(ortho_err)

            # Xuất ảnh debug 4 góc (Yêu cầu 7)
            banner = f"[{idx:02d}] {doc_type.upper()} | {m} | AR:{ar:.2f} (dAR:{ar_err:.2f}) | dAngle:{ortho_err:.1f}deg | {rect_latency_ms:.1f}ms"
            vis_corners = rectifier.draw_corners(img, rect_res.corners, title=banner)
            cv2.imwrite(str(corners_debug_dir / f"{idx:02d}_{doc_type}_corners_{filename}"), vis_corners)

            # Lưu ảnh nắn phẳng
            cv2.imwrite(str(corners_debug_dir / f"{idx:02d}_{doc_type}_warped_{filename}"), rect_res.rectified_image)

        else:
            # OpenCV Corner Detection THẤT BẠI
            corner_metrics["corners_failed"] += 1
            cause = analyze_failure_cause(yolo_raw_crop, bbox, (h_orig, w_orig))

            fail_info = {
                "index": idx,
                "filename": filename,
                "doc_type": doc_type,
                "yolo_bbox": list(bbox),
                "method_attempted": m,
                "error": rect_res.error or "No valid 4-corner quad found",
                "cause": cause,
            }
            failure_cases.append(fail_info)

            # Xuất ảnh debug cho ca thất bại (Yêu cầu 11)
            fail_vis = img.copy()
            cv2.rectangle(fail_vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(
                fail_vis,
                f"FAILED: {cause}",
                (bbox[0], max(25, bbox[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(failures_dir / f"fail_{idx:02d}_{doc_type}_{filename}"), fail_vis)

        # -------------------------------------------------------------------
        # Bước 3: So sánh đối đầu OCR (Trước vs Sau Rectification)
        # -------------------------------------------------------------------
        # Pipeline 1: Trước Rectification (Chỉ dùng YOLO Raw Crop)
        t_ocr1_start = time.perf_counter()
        try:
            fields_before = extract_baseline_yolo_crop(yolo_raw_crop, doc_type)
        except Exception as e:
            logger.debug(f"OCR Before failed on {filename}: {e}")
            fields_before = {}
        t_ocr1_end = time.perf_counter()
        ocr_before["latencies_sec"].append(t_ocr1_end - t_ocr1_start)

        # Pipeline 2: Sau Rectification (Dùng ảnh nắn phối cảnh hoặc fallback)
        t_ocr2_start = time.perf_counter()
        # Chạy orientation nếu cần và extract bằng calibrated extractor
        orient_res = correct_orientation(rect_res.rectified_image, card_type=doc_type)
        upright_img = orient_res.image
        try:
            if doc_type == "cccd":
                fields_after, _ = extract_calibrated_cccd_fields_with_raw(upright_img, fast_mode=True)
            else:
                fields_after, _ = extract_calibrated_gplx_fields_with_raw(upright_img, fast_mode=True)
        except Exception as e:
            logger.debug(f"OCR After failed on {filename}: {e}")
            fields_after = {}
        t_ocr2_end = time.perf_counter()
        ocr_after["latencies_sec"].append(t_ocr2_end - t_ocr2_start)

        # Đánh giá chỉ số từng trường
        for fld, val in fields_before.items():
            ocr_before["total_fields"] += 1
            f_stats = ocr_before["field_stats"].setdefault(fld, {"total": 0, "extracted": 0, "valid": 0})
            f_stats["total"] += 1
            if val:
                ocr_before["extracted"] += 1
                f_stats["extracted"] += 1
                validator = validators.get(fld, lambda v: bool(v))
                if validator(val):
                    ocr_before["valid"] += 1
                    f_stats["valid"] += 1

        for fld, val in fields_after.items():
            ocr_after["total_fields"] += 1
            f_stats = ocr_after["field_stats"].setdefault(fld, {"total": 0, "extracted": 0, "valid": 0})
            f_stats["total"] += 1
            if val:
                ocr_after["extracted"] += 1
                f_stats["extracted"] += 1
                validator = validators.get(fld, lambda v: bool(v))
                if validator(val):
                    ocr_after["valid"] += 1
                    f_stats["valid"] += 1

        logger.info(
            f"[{idx:02d}/{len(selected_images)}] {filename[:28]} | Doc={doc_type} | "
            f"Corners={'OK' if is_corner_ok else 'FAIL'} ({m}) | "
            f"OCR Valid: Before={sum(1 for k, v in fields_before.items() if validators.get(k, lambda x: bool(x))(v))}/{len(fields_before)} "
            f"-> After={sum(1 for k, v in fields_after.items() if validators.get(k, lambda x: bool(x))(v))}/{len(fields_after)}"
        )

    # -----------------------------------------------------------------------
    # Bước 4: Tổng hợp báo cáo định lượng
    # -----------------------------------------------------------------------
    det_rate = (corner_metrics["corners_success"] / corner_metrics["total_images"]) * 100.0
    fail_rate = (corner_metrics["corners_failed"] / corner_metrics["total_images"]) * 100.0

    mean_aspect = float(np.mean(corner_metrics["aspect_ratios"])) if corner_metrics["aspect_ratios"] else 0.0
    mean_aspect_err = float(np.mean(corner_metrics["aspect_errors"])) if corner_metrics["aspect_errors"] else 0.0
    mean_ortho_err = float(np.mean(corner_metrics["orthogonality_errors"])) if corner_metrics["orthogonality_errors"] else 0.0
    mean_latency_ms = float(np.mean(corner_metrics["latencies_ms"])) if corner_metrics["latencies_ms"] else 0.0

    ocr_before_ext_rate = (ocr_before["extracted"] / max(1, ocr_before["total_fields"])) * 100.0
    ocr_before_val_rate = (ocr_before["valid"] / max(1, ocr_before["total_fields"])) * 100.0
    ocr_before_avg_lat = float(np.mean(ocr_before["latencies_sec"])) if ocr_before["latencies_sec"] else 0.0

    ocr_after_ext_rate = (ocr_after["extracted"] / max(1, ocr_after["total_fields"])) * 100.0
    ocr_after_val_rate = (ocr_after["valid"] / max(1, ocr_after["total_fields"])) * 100.0
    ocr_after_avg_lat = float(np.mean(ocr_after["latencies_sec"])) if ocr_after["latencies_sec"] else 0.0

    report = {
        "benchmark_metadata": {
            "stage": "3.3",
            "name": "YOLOv8n Detection + OpenCV Document Corner Detection",
            "total_samples": len(selected_images),
            "cccd_samples": len(selected_cccd),
            "gplx_samples": len(selected_gplx),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "corner_detection_results": {
            "total_images": corner_metrics["total_images"],
            "yolo_detected": corner_metrics["yolo_detected"],
            "corners_success": corner_metrics["corners_success"],
            "corners_failed": corner_metrics["corners_failed"],
            "success_rate_percent": round(det_rate, 2),
            "failure_rate_percent": round(fail_rate, 2),
            "methods_breakdown": corner_metrics["methods"],
            "localization_error": {
                "mean_aspect_ratio": round(mean_aspect, 4),
                "ideal_id1_aspect_ratio": round(ISO_ID1_ASPECT_RATIO, 4),
                "mean_aspect_ratio_error": round(mean_aspect_err, 4),
                "mean_orthogonality_deviation_degrees": round(mean_ortho_err, 2),
            },
            "average_latency_ms": round(mean_latency_ms, 2),
        },
        "ocr_comparison": {
            "before_rectification_yolo_crop": {
                "field_extraction_rate_percent": round(ocr_before_ext_rate, 2),
                "field_validation_rate_percent": round(ocr_before_val_rate, 2),
                "average_latency_sec": round(ocr_before_avg_lat, 2),
                "field_level": ocr_before["field_stats"],
            },
            "after_rectification_opencv_warp": {
                "field_extraction_rate_percent": round(ocr_after_ext_rate, 2),
                "field_validation_rate_percent": round(ocr_after_val_rate, 2),
                "average_latency_sec": round(ocr_after_avg_lat, 2),
                "field_level": ocr_after["field_stats"],
            },
            "net_gain": {
                "extraction_rate_gain_percent": round(ocr_after_ext_rate - ocr_before_ext_rate, 2),
                "validation_rate_gain_percent": round(ocr_after_val_rate - ocr_before_val_rate, 2),
            },
        },
        "failure_analysis": {
            "total_failures": len(failure_cases),
            "cases": failure_cases,
        },
    }

    # Lưu file kết quả JSON
    report_json_path = out_dir / "benchmark_stage3_3_results.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"\n=======================================================")
    logger.info(f"BÁO CÁO TỔNG KẾT BENCHMARK GIAI ĐOẠN 3.3")
    logger.info(f"=======================================================")
    logger.info(f"Tổng số ảnh kiểm thử: {corner_metrics['total_images']}")
    logger.info(f"Tỷ lệ tìm đủ 4 góc thực tế: {det_rate:.1f}% ({corner_metrics['corners_success']}/{corner_metrics['total_images']})")
    logger.info(f"Tỷ lệ thất bại OpenCV: {fail_rate:.1f}% ({corner_metrics['corners_failed']}/{corner_metrics['total_images']})")
    logger.info(f"Sai số trực giao góc trung bình: {mean_ortho_err:.2f}°")
    logger.info(f"Sai số tỷ lệ khung hình ID-1: {mean_aspect_err:.4f}")
    logger.info(f"Thời gian định vị góc trung bình: {mean_latency_ms:.1f} ms")
    logger.info(f"--- SO SÁNH ĐỐI ĐẦU OCR ---")
    logger.info(f"Field Extraction: Trước = {ocr_before_ext_rate:.2f}% | Sau = {ocr_after_ext_rate:.2f}% (Tăng {ocr_after_ext_rate - ocr_before_ext_rate:+.2f}%)")
    logger.info(f"Field Validation: Trước = {ocr_before_val_rate:.2f}% | Sau = {ocr_after_val_rate:.2f}% (Tăng {ocr_after_val_rate - ocr_before_val_rate:+.2f}%)")
    logger.info(f"File kết quả JSON: {report_json_path}")
    logger.info(f"=======================================================\n")

    return report


if __name__ == "__main__":
    run_benchmark_stage3_3()
