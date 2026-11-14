"""Script benchmark so sánh pipeline Fixed ROI (Baseline) và YOLO ROI (YOLO -> Crop -> OCR).

Phục vụ Task: YOLOv8n Post-Training Audit & Integration Benchmark
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


def extract_cccd_all(image: np.ndarray) -> dict[str, str | None]:
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


def extract_gplx_all(image: np.ndarray) -> dict[str, str | None]:
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


def run_benchmark(
    sample_limit: int = 10,
    output_dir: str | Path = "debug_output/benchmark_integration",
) -> dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    test_img_dir = PROJECT_ROOT / "datasets" / "images" / "test"
    all_test_imgs = sorted([f for f in test_img_dir.glob("*.jpg")])

    cccd_imgs = [f for f in all_test_imgs if f.name.startswith("cccd_")]
    gplx_imgs = [f for f in all_test_imgs if f.name.startswith("gplx_")]

    # Lấy mẫu cân bằng CCCD và GPLX
    half = sample_limit // 2
    selected_imgs = cccd_imgs[:half] + gplx_imgs[:half]
    logger.info(f"Đã chọn {len(selected_imgs)} ảnh test để benchmark (CCCD: {len(cccd_imgs[:half])}, GPLX: {len(gplx_imgs[:half])}).")

    detector = DocumentDetector(device="cpu")

    results = []

    fixed_total_fields = 0
    fixed_success_fields = 0
    fixed_total_time = 0.0

    yolo_total_fields = 0
    yolo_success_fields = 0
    yolo_total_time = 0.0
    yolo_det_time = 0.0

    for idx, img_path in enumerate(selected_imgs, 1):
        is_cccd = img_path.name.startswith("cccd_")
        target_doc = "cccd" if is_cccd else "gplx"
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        logger.info(f"[{idx}/{len(selected_imgs)}] Đang kiểm thử: {img_path.name} ({w}x{h}, loại: {target_doc})")

        # 1. Pipeline 1: Fixed ROI (Baseline)
        t0 = time.perf_counter()
        if is_cccd:
            fixed_fields = extract_cccd_all(img)
        else:
            fixed_fields = extract_gplx_all(img)
        t_fixed = time.perf_counter() - t0

        fixed_valid_count = sum(1 for v in fixed_fields.values() if v is not None and str(v).strip())
        fixed_total_count = len(fixed_fields)

        fixed_total_fields += fixed_total_count
        fixed_success_fields += fixed_valid_count
        fixed_total_time += t_fixed

        # 2. Pipeline 2: YOLO ROI (YOLO -> Crop -> OCR)
        t0_det = time.perf_counter()
        det_result = detector.detect(img)
        t_det = time.perf_counter() - t0_det

        yolo_fields = {}
        t_ocr = 0.0
        crop_success = False
        crop_img = None
        card_class_pred = None
        conf = 0.0

        if det_result.has_detection and det_result.top_detection:
            top_det = det_result.top_detection
            card_class_pred = top_det.class_name
            conf = top_det.confidence
            x1, y1, x2, y2 = top_det.bbox_xyxy

            # Crop ảnh theo bbox
            crop_img = img[y1:y2, x1:x2]
            if crop_img.size > 0:
                crop_success = True
                t0_ocr = time.perf_counter()
                if card_class_pred == "cccd":
                    yolo_fields = extract_cccd_all(crop_img)
                else:
                    yolo_fields = extract_gplx_all(crop_img)
                t_ocr = time.perf_counter() - t0_ocr

        t_yolo_total = t_det + t_ocr
        yolo_valid_count = sum(1 for v in yolo_fields.values() if v is not None and str(v).strip())
        yolo_total_count = len(yolo_fields) if yolo_fields else (8 if is_cccd else 7)

        yolo_total_fields += yolo_total_count
        yolo_success_fields += yolo_valid_count
        yolo_total_time += t_yolo_total
        yolo_det_time += t_det

        # Lưu ảnh minh họa crop và bbox
        annotated = detector.annotate_image(img, det_result.detections)
        cv2.imwrite(str(out_path / f"det_{img_path.stem}.jpg"), annotated)
        if crop_img is not None and crop_img.size > 0:
            cv2.imwrite(str(out_path / f"crop_{img_path.stem}.jpg"), crop_img)

        sample_res = {
            "image": img_path.name,
            "target_type": target_doc,
            "predicted_type": card_class_pred,
            "confidence": round(conf, 4),
            "yolo_detected": det_result.has_detection,
            "fixed_roi": {
                "time_sec": round(t_fixed, 3),
                "valid_fields": fixed_valid_count,
                "total_fields": fixed_total_count,
                "fields": fixed_fields,
            },
            "yolo_roi": {
                "det_time_sec": round(t_det, 3),
                "ocr_time_sec": round(t_ocr, 3),
                "total_time_sec": round(t_yolo_total, 3),
                "valid_fields": yolo_valid_count,
                "total_fields": yolo_total_count,
                "fields": yolo_fields,
            },
            "improvement": yolo_valid_count - fixed_valid_count,
        }
        results.append(sample_res)

    summary = {
        "total_samples": len(results),
        "fixed_pipeline": {
            "avg_time_sec": round(fixed_total_time / len(results), 3) if results else 0,
            "field_success_rate": round(fixed_success_fields / fixed_total_fields * 100, 2) if fixed_total_fields else 0,
            "total_valid_fields": fixed_success_fields,
            "total_possible_fields": fixed_total_fields,
        },
        "yolo_pipeline": {
            "avg_det_time_sec": round(yolo_det_time / len(results), 3) if results else 0,
            "avg_total_time_sec": round(yolo_total_time / len(results), 3) if results else 0,
            "field_success_rate": round(yolo_success_fields / yolo_total_fields * 100, 2) if yolo_total_fields else 0,
            "total_valid_fields": yolo_success_fields,
            "total_possible_fields": yolo_total_fields,
        },
        "samples": results,
    }

    report_json_path = out_path / "benchmark_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Hoàn thành benchmark! Báo cáo lưu tại: {report_json_path}")
    return summary


if __name__ == "__main__":
    summary = run_benchmark(sample_limit=8)
    print("\n" + "=" * 50)
    print("TỔNG HỢP KẾT QUẢ BENCHMARK INTEGRATION:")
    print(f"- Số ảnh thử nghiệm: {summary['total_samples']}")
    print(f"- Fixed ROI Baseline: Tỷ lệ trường đọc được = {summary['fixed_pipeline']['field_success_rate']}% | Thời gian TB = {summary['fixed_pipeline']['avg_time_sec']}s")
    print(f"- YOLO ROI Pipeline:  Tỷ lệ trường đọc được = {summary['yolo_pipeline']['field_success_rate']}% | Thời gian TB = {summary['yolo_pipeline']['avg_total_time_sec']}s (Detection: {summary['yolo_pipeline']['avg_det_time_sec']}s)")
    print("=" * 50)
