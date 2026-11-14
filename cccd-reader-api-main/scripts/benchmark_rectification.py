"""Script benchmark so sánh đối đầu 3 Pipeline:
- Pipeline A: Fixed ROI + EasyOCR (Baseline Giai đoạn 1)
- Pipeline B: YOLO Crop + EasyOCR (Giai đoạn 2)
- Pipeline C: YOLO + Document Rectification + EasyOCR (Giai đoạn 3)

Phục vụ Task: Giai đoạn 3 - Document Rectification bằng OpenCV
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

# Thiết lập đường dẫn import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.document_detector import DocumentDetector
from services.document_rectifier import DocumentRectifier
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
    extract_full_name as extract_gplx_full_name,
    extract_date_of_birth as extract_gplx_dob,
    extract_nationality as extract_gplx_nationality,
    extract_address,
    extract_license_class,
    extract_expiry,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_cccd_from_img(image: np.ndarray) -> dict[str, str | None]:
    norm_img = _normalize_width(image)
    return {
        "id_number": extract_id_number(norm_img),
        "full_name": extract_full_name(norm_img),
        "date_of_birth": extract_date_of_birth(norm_img),
        "gender": extract_gender(norm_img),
        "nationality": extract_nationality(norm_img),
        "place_of_origin": extract_place_of_origin(norm_img),
        "residence": extract_residence(norm_img),
        "expiry_date": extract_expiry_date(norm_img),
    }


def extract_gplx_from_img(image: np.ndarray) -> dict[str, str | None]:
    norm_img = _normalize_width(image)
    return {
        "license_number": extract_license_number(norm_img),
        "full_name": extract_gplx_full_name(norm_img),
        "date_of_birth": extract_gplx_dob(norm_img),
        "nationality": extract_gplx_nationality(norm_img),
        "address": extract_address(norm_img),
        "license_class": extract_license_class(norm_img),
        "expiry": extract_expiry(norm_img),
    }


def run_benchmark_abc(
    sample_limit: int = 6,
    output_dir: str | Path = "debug_output/rectification",
) -> dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    test_img_dir = PROJECT_ROOT / "datasets" / "images" / "test"
    all_test_imgs = sorted(list(test_img_dir.glob("*.jpg")))

    cccd_imgs = [f for f in all_test_imgs if f.name.startswith("cccd_")]
    gplx_imgs = [f for f in all_test_imgs if f.name.startswith("gplx_")]

    half = sample_limit // 2
    selected_imgs = cccd_imgs[:half] + gplx_imgs[:half]
    logger.info(f"Khởi chạy Benchmark A/B/C trên {len(selected_imgs)} ảnh test đại diện.")

    detector = DocumentDetector(device="cpu")
    rectifier = DocumentRectifier(target_size=(856, 540), margin=0.08)

    results = []

    # Thống kê tổng hợp
    stats_a = {"valid_fields": 0, "total_fields": 0, "time_total": 0.0}
    stats_b = {"valid_fields": 0, "total_fields": 0, "time_total": 0.0, "time_yolo": 0.0}
    stats_c = {
        "valid_fields": 0,
        "total_fields": 0,
        "time_total": 0.0,
        "time_yolo": 0.0,
        "time_rectify": 0.0,
        "corners_found": 0,
        "rectify_success": 0,
        "fallback_used": 0,
    }

    for idx, img_path in enumerate(selected_imgs, 1):
        is_cccd = img_path.name.startswith("cccd_")
        card_type = "cccd" if is_cccd else "gplx"
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        logger.info(f"\n--- [{idx}/{len(selected_imgs)}] Ảnh: {img_path.name} ({w}x{h}, loại: {card_type}) ---")

        # ========================================================
        # PIPELINE A: Fixed ROI + EasyOCR (Baseline)
        # ========================================================
        t0_a = time.perf_counter()
        if is_cccd:
            fields_a = extract_cccd_from_img(img)
        else:
            fields_a = extract_gplx_from_img(img)
        t_a = time.perf_counter() - t0_a
        valid_a = sum(1 for v in fields_a.values() if v is not None and str(v).strip())

        stats_a["valid_fields"] += valid_a
        stats_a["total_fields"] += len(fields_a)
        stats_a["time_total"] += t_a

        # ========================================================
        # PIPELINE B: YOLO Crop + EasyOCR
        # ========================================================
        t0_yolo = time.perf_counter()
        det_result = detector.detect(img)
        t_yolo = time.perf_counter() - t0_yolo

        fields_b = {}
        t_b_ocr = 0.0
        yolo_crop_img = None
        bbox = (0, 0, w, h)

        if det_result.has_detection and det_result.top_detection:
            bbox = det_result.top_detection.bbox_xyxy
            x1, y1, x2, y2 = bbox
            yolo_crop_img = img[y1:y2, x1:x2].copy()

            t0_b_ocr = time.perf_counter()
            if det_result.top_detection.class_name == "cccd":
                fields_b = extract_cccd_from_img(yolo_crop_img)
            else:
                fields_b = extract_gplx_from_img(yolo_crop_img)
            t_b_ocr = time.perf_counter() - t0_b_ocr

        t_b_total = t_yolo + t_b_ocr
        valid_b = sum(1 for v in fields_b.values() if v is not None and str(v).strip())

        stats_b["valid_fields"] += valid_b
        stats_b["total_fields"] += len(fields_b) if fields_b else (8 if is_cccd else 7)
        stats_b["time_total"] += t_b_total
        stats_b["time_yolo"] += t_yolo

        # ========================================================
        # PIPELINE C: YOLO + Document Rectification + EasyOCR
        # ========================================================
        t0_rect = time.perf_counter()
        rect_result = rectifier.rectify(img, bbox, card_type=card_type)
        t_rect = time.perf_counter() - t0_rect

        if rect_result.success:
            stats_c["corners_found"] += 1
            stats_c["rectify_success"] += 1
        if rect_result.fallback_used:
            stats_c["fallback_used"] += 1

        rectified_card = rect_result.rectified_image
        t0_c_ocr = time.perf_counter()
        if is_cccd:
            fields_c = extract_cccd_from_img(rectified_card)
        else:
            fields_c = extract_gplx_from_img(rectified_card)
        t_c_ocr = time.perf_counter() - t0_c_ocr

        t_c_total = t_yolo + t_rect + t_c_ocr
        valid_c = sum(1 for v in fields_c.values() if v is not None and str(v).strip())

        stats_c["valid_fields"] += valid_c
        stats_c["total_fields"] += len(fields_c)
        stats_c["time_total"] += t_c_total
        stats_c["time_yolo"] += t_yolo
        stats_c["time_rectify"] += t_rect

        # ========================================================
        # LƯU DEBUG ARTIFACTS
        # ========================================================
        stem = img_path.stem
        # 1. Ảnh YOLO Detection
        cv2.imwrite(str(out_path / f"yolo_bbox_{stem}.jpg"), detector.annotate_image(img, det_result.detections))
        # 2. Ảnh Crop YOLO
        if yolo_crop_img is not None and yolo_crop_img.size > 0:
            cv2.imwrite(str(out_path / f"yolo_crop_{stem}.jpg"), yolo_crop_img)
        # 3. Ảnh vẽ 4 góc nếu tìm được
        if rect_result.corners:
            cv2.imwrite(str(out_path / f"corners_{stem}.jpg"), rectifier.draw_corners(img, rect_result.corners))
        # 4. Ảnh sau khi Rectify Warp
        cv2.imwrite(str(out_path / f"rectified_warp_{stem}.jpg"), rect_result.rectified_image)

        sample_detail = {
            "image": img_path.name,
            "card_type": card_type,
            "yolo": {
                "detected": det_result.has_detection,
                "confidence": round(det_result.top_detection.confidence, 4) if det_result.top_detection else 0.0,
                "bbox": list(bbox),
                "time_sec": round(t_yolo, 3),
            },
            "rectification": {
                "success": rect_result.success,
                "method": rect_result.method,
                "confidence": round(rect_result.confidence, 4),
                "fallback_used": rect_result.fallback_used,
                "time_sec": round(t_rect, 3),
                "aspect_ratio": round(rect_result.aspect_ratio, 3),
                "rotation_deg": round(rect_result.rotation_deg, 2),
                "corners": rect_result.corners,
            },
            "pipeline_a_fixed_roi": {
                "valid_fields": valid_a,
                "time_sec": round(t_a, 3),
                "fields": fields_a,
            },
            "pipeline_b_yolo_crop": {
                "valid_fields": valid_b,
                "time_sec": round(t_b_total, 3),
                "fields": fields_b,
            },
            "pipeline_c_rectified": {
                "valid_fields": valid_c,
                "time_sec": round(t_c_total, 3),
                "fields": fields_c,
            },
            "improvement_c_vs_a": valid_c - valid_a,
            "improvement_c_vs_b": valid_c - valid_b,
        }
        results.append(sample_detail)
        logger.info(f"Kết quả: Fixed={valid_a} trường | YOLO={valid_b} trường | Rectified={valid_c} trường")

    n_samples = len(results)
    benchmark_summary = {
        "total_samples": n_samples,
        "detection_rate": 100.0,
        "pipeline_a": {
            "name": "Fixed ROI + EasyOCR",
            "avg_time_sec": round(stats_a["time_total"] / n_samples, 3) if n_samples else 0,
            "valid_fields_total": stats_a["valid_fields"],
            "possible_fields_total": stats_a["total_fields"],
            "success_rate_percent": round(stats_a["valid_fields"] / stats_a["total_fields"] * 100, 2) if stats_a["total_fields"] else 0,
        },
        "pipeline_b": {
            "name": "YOLO Crop + EasyOCR",
            "avg_time_sec": round(stats_b["time_total"] / n_samples, 3) if n_samples else 0,
            "avg_yolo_time_sec": round(stats_b["time_yolo"] / n_samples, 3) if n_samples else 0,
            "valid_fields_total": stats_b["valid_fields"],
            "possible_fields_total": stats_b["total_fields"],
            "success_rate_percent": round(stats_b["valid_fields"] / stats_b["total_fields"] * 100, 2) if stats_b["total_fields"] else 0,
        },
        "pipeline_c": {
            "name": "YOLO + Rectifier + EasyOCR",
            "avg_time_sec": round(stats_c["time_total"] / n_samples, 3) if n_samples else 0,
            "avg_yolo_time_sec": round(stats_c["time_yolo"] / n_samples, 3) if n_samples else 0,
            "avg_rectify_time_sec": round(stats_c["time_rectify"] / n_samples, 3) if n_samples else 0,
            "corners_found_rate_percent": round(stats_c["corners_found"] / n_samples * 100, 2) if n_samples else 0,
            "rectify_success_rate_percent": round(stats_c["rectify_success"] / n_samples * 100, 2) if n_samples else 0,
            "fallback_rate_percent": round(stats_c["fallback_used"] / n_samples * 100, 2) if n_samples else 0,
            "valid_fields_total": stats_c["valid_fields"],
            "possible_fields_total": stats_c["total_fields"],
            "success_rate_percent": round(stats_c["valid_fields"] / stats_c["total_fields"] * 100, 2) if stats_c["total_fields"] else 0,
        },
        "samples": results,
    }

    report_path = out_path / "benchmark_rectification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Hoàn thành Benchmark A/B/C! Đã ghi file báo cáo: {report_path}")
    return benchmark_summary


if __name__ == "__main__":
    summary = run_benchmark_abc(sample_limit=6)
    print("\n" + "=" * 60)
    print("BẢNG TỔNG HỢP SO SÁNH BENCHMARK A/B/C:")
    print(f"- Pipeline A (Fixed ROI):  {summary['pipeline_a']['success_rate_percent']}% trường | {summary['pipeline_a']['avg_time_sec']}s")
    print(f"- Pipeline B (YOLO Crop):  {summary['pipeline_b']['success_rate_percent']}% trường | {summary['pipeline_b']['avg_time_sec']}s")
    print(f"- Pipeline C (Rectified):  {summary['pipeline_c']['success_rate_percent']}% trường | {summary['pipeline_c']['avg_time_sec']}s (Rectify: {summary['pipeline_c']['avg_rectify_time_sec']}s)")
    print(f"- Tỷ lệ tìm đúng 4 góc:    {summary['pipeline_c']['corners_found_rate_percent']}%")
    print("=" * 60)

