"""Benchmark so sánh 4 Pipeline A / B / C / D (Giai đoạn 3.1).

Các pipeline so sánh:
- Pipeline A: Fixed ROI + EasyOCR (Baseline Giai đoạn 1)
- Pipeline B: YOLO Crop + EasyOCR (Giai đoạn 2)
- Pipeline C: YOLO + Document Rectifier + EasyOCR (Giai đoạn 3)
- Pipeline D: YOLO + Document Rectifier + Orientation Correction + Calibrated ROIs + Quality Fallback (Giai đoạn 3.1)

Xuất kết quả:
- debug_output/stage3_1/benchmark_stage3_1_report.json
- debug_output/stage3_1/*.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.document_detector import DocumentDetector
from services.document_rectifier import DocumentRectifier
from services.document_orientation import correct_orientation
from services.calibrated_rois import (
    extract_calibrated_cccd_fields,
    extract_calibrated_gplx_fields,
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_cccd_baseline(img: np.ndarray) -> dict[str, str | None]:
    norm = _normalize_width(img)
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


def extract_gplx_baseline(img: np.ndarray) -> dict[str, str | None]:
    norm = _normalize_width(img)
    return {
        "license_number": extract_license_number(norm),
        "full_name": extract_gplx_name(norm),
        "date_of_birth": extract_gplx_dob(norm),
        "nationality": extract_gplx_nat(norm),
        "address": extract_address(norm),
        "license_class": extract_license_class(norm),
        "expiry": extract_expiry(norm),
    }


def run_benchmark(num_samples: int = 6):
    out_dir = PROJECT_ROOT / "debug_output" / "stage3_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_images_dir = PROJECT_ROOT / "datasets" / "images" / "test"
    model_path = PROJECT_ROOT / "models" / "yolov8n_document.pt"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "runs" / "detect" / "smoke_test_10e" / "weights" / "best.pt"

    detector = DocumentDetector(model_path=str(model_path), conf_threshold=0.25)
    rectifier = DocumentRectifier(margin=0.08)

    # Chọn tập mẫu đại diện đa dạng: CCCD thẳng, CCCD nghiêng, CCCD ngược 180°, GPLX thẳng, GPLX nghiêng, GPLX dọc
    all_cccd = sorted(test_images_dir.glob("cccd_*.jpg"))
    all_gplx = sorted(test_images_dir.glob("gplx_*.jpg"))

    half = max(1, num_samples // 2)
    sample_paths = all_cccd[:half] + all_gplx[:half]

    logger.info(f"Bắt đầu Benchmark Giai đoạn 3.1 trên {len(sample_paths)} ảnh mẫu.")

    stats = {
        "total_samples": len(sample_paths),
        "pipeline_a": {"name": "Fixed ROI + EasyOCR", "valid_fields": 0, "total_fields": 0, "times": []},
        "pipeline_b": {"name": "YOLO Crop + EasyOCR", "valid_fields": 0, "total_fields": 0, "times": []},
        "pipeline_c": {"name": "YOLO + Rectifier + EasyOCR", "valid_fields": 0, "total_fields": 0, "times": []},
        "pipeline_d": {
            "name": "YOLO + Rect + Orient + CalibROI + Fallback",
            "valid_fields": 0,
            "total_fields": 0,
            "times": [],
            "fallback_count": 0,
            "rotations": {0: 0, 90: 0, 180: 0, 270: 0},
        },
        "per_field_stats": {
            "cccd": {},
            "gplx": {},
        },
        "samples": [],
    }

    for idx, p in enumerate(sample_paths):
        img = cv2.imread(str(p))
        if img is None:
            continue

        prefix = f"{idx+1:02d}_{p.stem[:22]}"
        logger.info(f"[{idx+1}/{len(sample_paths)}] Đang xử lý: {p.name}")

        # -------------------------------------------------------------
        # 1. Pipeline A: Fixed ROI
        # -------------------------------------------------------------
        is_cccd_name = "cccd" in p.name.lower()
        t0 = time.perf_counter()
        fields_a = extract_cccd_baseline(img) if is_cccd_name else extract_gplx_baseline(img)
        time_a = time.perf_counter() - t0
        valid_a = sum(1 for v in fields_a.values() if v is not None)
        total_f = len(fields_a)

        stats["pipeline_a"]["valid_fields"] += valid_a
        stats["pipeline_a"]["total_fields"] += total_f
        stats["pipeline_a"]["times"].append(time_a)

        # -------------------------------------------------------------
        # YOLO Detection chung cho B, C, D
        # -------------------------------------------------------------
        t0_yolo = time.perf_counter()
        det_res = detector.detect(img)
        time_yolo = time.perf_counter() - t0_yolo

        if not det_res.has_detection:
            logger.warning(f"YOLO không phát hiện được thẻ trên {p.name}")
            continue

        det = det_res.top_detection
        card_type = det.class_name
        x1, y1, x2, y2 = det.bbox_xyxy
        yolo_crop = img[y1:y2, x1:x2].copy()

        # -------------------------------------------------------------
        # 2. Pipeline B: YOLO Crop + EasyOCR Baseline
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        fields_b = extract_cccd_baseline(yolo_crop) if card_type == "cccd" else extract_gplx_baseline(yolo_crop)
        time_b = (time.perf_counter() - t0) + time_yolo
        valid_b = sum(1 for v in fields_b.values() if v is not None)

        stats["pipeline_b"]["valid_fields"] += valid_b
        stats["pipeline_b"]["total_fields"] += total_f
        stats["pipeline_b"]["times"].append(time_b)

        # -------------------------------------------------------------
        # 3. Pipeline C: YOLO + Rectifier + EasyOCR Baseline
        # -------------------------------------------------------------
        t0_rect = time.perf_counter()
        rect_res = rectifier.rectify(img, det.bbox_xyxy, card_type=card_type)
        time_rect = time.perf_counter() - t0_rect

        warped_c = rect_res.rectified_image
        t0 = time.perf_counter()
        fields_c = extract_cccd_baseline(warped_c) if card_type == "cccd" else extract_gplx_baseline(warped_c)
        time_c = (time.perf_counter() - t0) + time_yolo + time_rect
        valid_c = sum(1 for v in fields_c.values() if v is not None)

        stats["pipeline_c"]["valid_fields"] += valid_c
        stats["pipeline_c"]["total_fields"] += total_f
        stats["pipeline_c"]["times"].append(time_c)

        # -------------------------------------------------------------
        # 4. Pipeline D: YOLO + Rectifier + Orientation + Calibrated ROI + Quality Fallback
        # -------------------------------------------------------------
        t0_d = time.perf_counter()
        use_fallback_b = False
        fallback_reason = None

        # Kiểm tra chất lượng hình học
        if not rect_res.success or rect_res.fallback_used:
            use_fallback_b = True
            fallback_reason = "Rectification failed or fallback used"
        elif rect_res.aspect_ratio < 1.25 or rect_res.aspect_ratio > 2.15:
            use_fallback_b = True
            fallback_reason = f"Aspect ratio abnormal: {rect_res.aspect_ratio:.2f}"
        elif rect_res.confidence < 0.40:
            use_fallback_b = True
            fallback_reason = f"Low rectification confidence: {rect_res.confidence:.2f}"

        # Nếu hình học tốt -> Orientation Correction
        t0_orient = time.perf_counter()
        if not use_fallback_b:
            orient_res = correct_orientation(rect_res.rectified_image, card_type=card_type)
            time_orient = time.perf_counter() - t0_orient
            stats["pipeline_d"]["rotations"][orient_res.rotation_deg] = (
                stats["pipeline_d"]["rotations"].get(orient_res.rotation_deg, 0) + 1
            )
            final_img_d = orient_res.image
        else:
            time_orient = 0.0
            orient_res = None
            final_img_d = yolo_crop

        # Trích xuất trường theo ROI phù hợp
        t0_ocr_d = time.perf_counter()
        if not use_fallback_b:
            if card_type == "cccd":
                fields_d = extract_calibrated_cccd_fields(final_img_d)
            else:
                fields_d = extract_calibrated_gplx_fields(final_img_d)
        else:
            stats["pipeline_d"]["fallback_count"] += 1
            if card_type == "cccd":
                fields_d = extract_cccd_baseline(final_img_d)
            else:
                fields_d = extract_gplx_baseline(final_img_d)

        time_ocr_d = time.perf_counter() - t0_ocr_d
        time_d = time_yolo + time_rect + time_orient + time_ocr_d
        valid_d = sum(1 for v in fields_d.values() if v is not None)

        stats["pipeline_d"]["valid_fields"] += valid_d
        stats["pipeline_d"]["total_fields"] += total_f
        stats["pipeline_d"]["times"].append(time_d)

        # Cập nhật thống kê per-field
        for fld, val in fields_d.items():
            if fld not in stats["per_field_stats"][card_type]:
                stats["per_field_stats"][card_type][fld] = {"A": 0, "B": 0, "C": 0, "D": 0, "total": 0}
            stats["per_field_stats"][card_type][fld]["total"] += 1
            if fields_a.get(fld): stats["per_field_stats"][card_type][fld]["A"] += 1
            if fields_b.get(fld): stats["per_field_stats"][card_type][fld]["B"] += 1
            if fields_c.get(fld): stats["per_field_stats"][card_type][fld]["C"] += 1
            if fields_d.get(fld): stats["per_field_stats"][card_type][fld]["D"] += 1

        # Lưu ảnh trực quan
        cv2.imwrite(str(out_dir / f"{prefix}_warped_c.jpg"), warped_c)
        cv2.imwrite(str(out_dir / f"{prefix}_final_d.jpg"), final_img_d)

        sample_info = {
            "file": p.name,
            "card_type": card_type,
            "rotation_applied_d": orient_res.rotation_deg if orient_res else 0,
            "fallback_used_d": use_fallback_b,
            "fallback_reason_d": fallback_reason,
            "valid_counts": {"A": valid_a, "B": valid_b, "C": valid_c, "D": valid_d, "total": total_f},
            "latencies_sec": {
                "A": round(time_a, 2),
                "B": round(time_b, 2),
                "C": round(time_c, 2),
                "D": round(time_d, 2),
                "yolo": round(time_yolo, 3),
                "rectify": round(time_rect, 3),
                "orient": round(time_orient, 3),
                "ocr_d": round(time_ocr_d, 2),
            },
            "fields_d": fields_d,
            "fields_b": fields_b,
        }
        stats["samples"].append(sample_info)
        logger.info(
            f"  Kết quả: A={valid_a}/{total_f}, B={valid_b}/{total_f}, C={valid_c}/{total_f}, D={valid_d}/{total_f} "
            f"(Time D={time_d:.1f}s, Rot={orient_res.rotation_deg if orient_res else 0}°)"
        )

    # Tổng kết
    for p_key in ["pipeline_a", "pipeline_b", "pipeline_c", "pipeline_d"]:
        v = stats[p_key]["valid_fields"]
        tot = stats[p_key]["total_fields"]
        stats[p_key]["rate_percent"] = round((v / tot * 100) if tot > 0 else 0.0, 2)
        stats[p_key]["avg_time_sec"] = round(float(np.mean(stats[p_key]["times"])) if stats[p_key]["times"] else 0.0, 2)

    report_path = out_dir / "benchmark_stage3_1_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("================ BENCHMARK COMPLETED ================")
    logger.info(f"Pipeline A (Fixed ROI):  {stats['pipeline_a']['rate_percent']}% ({stats['pipeline_a']['avg_time_sec']}s)")
    logger.info(f"Pipeline B (YOLO Crop):  {stats['pipeline_b']['rate_percent']}% ({stats['pipeline_b']['avg_time_sec']}s)")
    logger.info(f"Pipeline C (Rectifier):  {stats['pipeline_c']['rate_percent']}% ({stats['pipeline_c']['avg_time_sec']}s)")
    logger.info(f"Pipeline D (Giai đoạn 3.1): {stats['pipeline_d']['rate_percent']}% ({stats['pipeline_d']['avg_time_sec']}s)")
    logger.info(f"Báo cáo chi tiết đã lưu tại: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=6)
    args = parser.parse_args()
    run_benchmark(num_samples=args.num_samples)

