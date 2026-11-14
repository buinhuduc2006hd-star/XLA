"""Benchmark toàn diện 5 Pipeline A / B / C / D / E (Giai đoạn 3.2).

5 Pipeline đánh giá:
- Pipeline A: Fixed ROI + EasyOCR (Baseline Giai đoạn 1)
- Pipeline B: YOLO Crop + EasyOCR (Giai đoạn 2)
- Pipeline C: YOLO + Rectifier + EasyOCR (Giai đoạn 3)
- Pipeline D: YOLO + Rect + Orient + CalibROI + Geometric Fallback (Giai đoạn 3.1)
- Pipeline E: Optimized OCR (Giai đoạn 3.2: CalibROI + Early Exit + Adaptive Prep + Field-Level Fallback)

Chỉ số đo lường toàn diện:
- Field Extraction Rate (%)
- Field Validation Rate (%)
- Exact-Match Accuracy (%)
- Character Error Rate (CER)
- Word Error Rate (WER)
- Document-Level Accuracy (%)
- Latency phân bổ: Average, Median, P95 (sec)
- Chi tiết Fallback: Tổng số fallback kích hoạt, số fallback thành công, tỷ lệ hồi phục.

Đầu ra:
- debug_output/stage3_2/benchmark_stage3_2_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

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
from services.adaptive_extractor import (
    extract_adaptive_pipeline_e,
    compute_cer,
    compute_wer,
    is_exact_match,
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


def run_benchmark():
    out_dir = PROJECT_ROOT / "debug_output" / "stage3_2"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = PROJECT_ROOT / "datasets" / "ground_truth_test_sample.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth: dict[str, dict[str, Any]] = json.load(f)

    test_images_dir = PROJECT_ROOT / "datasets" / "images" / "test"
    model_path = PROJECT_ROOT / "models" / "yolov8n_document.pt"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "runs" / "detect" / "smoke_test_10e" / "weights" / "best.pt"

    detector = DocumentDetector(model_path=str(model_path), conf_threshold=0.25)
    rectifier = DocumentRectifier(margin=0.08)

    sample_files = list(ground_truth.keys())
    logger.info(f"Bắt đầu Benchmark Giai đoạn 3.2 trên {len(sample_files)} ảnh mẫu có Ground Truth.")

    pipeline_keys = ["A", "B", "C", "D", "E"]
    pipeline_names = {
        "A": "Fixed ROI + EasyOCR (Stage 1)",
        "B": "YOLO Crop + EasyOCR (Stage 2)",
        "C": "YOLO + Rectifier + EasyOCR (Stage 3)",
        "D": "YOLO + Rect + Orient + CalibROI (Stage 3.1)",
        "E": "Optimized OCR + Adaptive Fallback (Stage 3.2)",
    }

    stats = {
        "total_samples": len(sample_files),
        "pipelines": {
            p: {
                "name": pipeline_names[p],
                "extracted_fields": 0,
                "validated_fields": 0,
                "exact_matches": 0,
                "total_fields": 0,
                "doc_exact_matches": 0,
                "cer_list": [],
                "wer_list": [],
                "latencies": [],
            }
            for p in pipeline_keys
        },
        "field_fallback_summary": {
            "total_triggered": 0,
            "total_recovered": 0,
            "per_field": {},
        },
        "per_field_accuracy": {
            "cccd": {},
            "gplx": {},
        },
        "sample_details": [],
    }

    for idx, fname in enumerate(sample_files):
        p = test_images_dir / fname
        if not p.exists():
            logger.warning(f"File {p} không tồn tại!")
            continue

        gt = ground_truth[fname]
        card_type = gt["card_type"]
        gt_fields = {k: v for k, v in gt.items() if k != "card_type"}
        total_card_fields = len(gt_fields)

        img = cv2.imread(str(p))
        if img is None:
            continue

        logger.info(f"[{idx+1}/{len(sample_files)}] Đang benchmark: {fname} ({card_type.upper()})")

        validators = CCCD_VALIDATORS if card_type == "cccd" else GPLX_VALIDATORS

        # -------------------------------------------------------------
        # 1. Pipeline A: Fixed ROI
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        fields_a = extract_cccd_baseline(img) if card_type == "cccd" else extract_gplx_baseline(img)
        lat_a = time.perf_counter() - t0

        # -------------------------------------------------------------
        # YOLO Detection (dùng chung cho B, C, D, E)
        # -------------------------------------------------------------
        t0_yolo = time.perf_counter()
        det_res = detector.detect(img)
        lat_yolo = time.perf_counter() - t0_yolo

        if not det_res.has_detection:
            logger.warning(f"YOLO không phát hiện được thẻ trên {fname}")
            continue

        det = det_res.top_detection
        x1, y1, x2, y2 = det.bbox_xyxy
        yolo_crop = img[y1:y2, x1:x2].copy()

        # -------------------------------------------------------------
        # 2. Pipeline B: YOLO Crop + Baseline
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        fields_b = extract_cccd_baseline(yolo_crop) if card_type == "cccd" else extract_gplx_baseline(yolo_crop)
        lat_b = (time.perf_counter() - t0) + lat_yolo

        # -------------------------------------------------------------
        # 3. Pipeline C: YOLO + Rectifier + Baseline
        # -------------------------------------------------------------
        t0_rect = time.perf_counter()
        rect_res = rectifier.rectify(img, det.bbox_xyxy, card_type=card_type)
        lat_rect = time.perf_counter() - t0_rect
        warped_c = rect_res.rectified_image

        t0 = time.perf_counter()
        fields_c = extract_cccd_baseline(warped_c) if card_type == "cccd" else extract_gplx_baseline(warped_c)
        lat_c = (time.perf_counter() - t0) + lat_yolo + lat_rect

        # -------------------------------------------------------------
        # 4. Pipeline D: YOLO + Rect + Orient + CalibROI (Giai đoạn 3.1)
        # -------------------------------------------------------------
        t0_orient = time.perf_counter()
        orient_res = correct_orientation(warped_c, card_type=card_type)
        lat_orient = time.perf_counter() - t0_orient
        final_img_d = orient_res.image

        t0 = time.perf_counter()
        if card_type == "cccd":
            fields_d = extract_calibrated_cccd_fields(final_img_d)
        else:
            fields_d = extract_calibrated_gplx_fields(final_img_d)
        lat_ocr_d = time.perf_counter() - t0
        lat_d = lat_yolo + lat_rect + lat_orient + lat_ocr_d

        # -------------------------------------------------------------
        # 5. Pipeline E: Optimized OCR + Adaptive Preprocessing + Field Fallback (Giai đoạn 3.2)
        # -------------------------------------------------------------
        t0_e = time.perf_counter()
        res_e = extract_adaptive_pipeline_e(final_img_d, yolo_crop, card_type=card_type)
        lat_ocr_e = time.perf_counter() - t0_e
        lat_e = lat_yolo + lat_rect + lat_orient + lat_ocr_e
        fields_e = res_e.fields

        # Cập nhật Fallback Statistics cho Pipeline E
        stats["field_fallback_summary"]["total_triggered"] += res_e.total_fallbacks
        stats["field_fallback_summary"]["total_recovered"] += res_e.successful_fallbacks
        for fld, used in res_e.fallback_used.items():
            if used:
                if fld not in stats["field_fallback_summary"]["per_field"]:
                    stats["field_fallback_summary"]["per_field"][fld] = {"triggered": 0, "recovered": 0}
                stats["field_fallback_summary"]["per_field"][fld]["triggered"] += 1
                if res_e.valid_flags[fld]:
                    stats["field_fallback_summary"]["per_field"][fld]["recovered"] += 1

        all_pipe_fields = {
            "A": fields_a,
            "B": fields_b,
            "C": fields_c,
            "D": fields_d,
            "E": fields_e,
        }
        all_pipe_lats = {
            "A": lat_a,
            "B": lat_b,
            "C": lat_c,
            "D": lat_d,
            "E": lat_e,
        }

        # Đánh giá chi tiết cho từng Pipeline trên từng trường
        sample_eval = {
            "file": fname,
            "card_type": card_type,
            "latencies": {k: round(v, 2) for k, v in all_pipe_lats.items()},
            "rotation_deg": orient_res.rotation_deg,
            "field_scores": {},
        }

        for p_key in pipeline_keys:
            p_fields = all_pipe_fields[p_key]
            stats["pipelines"][p_key]["latencies"].append(all_pipe_lats[p_key])
            stats["pipelines"][p_key]["total_fields"] += total_card_fields

            doc_exact = True
            for fld, ref_val in gt_fields.items():
                pred_val = p_fields.get(fld)

                # 1. Extraction Rate
                if pred_val is not None:
                    stats["pipelines"][p_key]["extracted_fields"] += 1

                # 2. Validation Rate
                v_func = validators.get(fld, lambda v: v is not None)
                if v_func(pred_val):
                    stats["pipelines"][p_key]["validated_fields"] += 1

                # 3. Exact Match
                em = is_exact_match(pred_val, ref_val)
                if em:
                    stats["pipelines"][p_key]["exact_matches"] += 1
                else:
                    doc_exact = False

                # 4. CER & WER
                c_err = compute_cer(pred_val, ref_val)
                w_err = compute_wer(pred_val, ref_val)
                stats["pipelines"][p_key]["cer_list"].append(c_err)
                stats["pipelines"][p_key]["wer_list"].append(w_err)

                # Per-field tracking
                if fld not in stats["per_field_accuracy"][card_type]:
                    stats["per_field_accuracy"][card_type][fld] = {
                        k: {"exact": 0, "valid": 0, "cer_sum": 0.0, "total": 0} for k in pipeline_keys
                    }
                stats["per_field_accuracy"][card_type][fld][p_key]["total"] += 1
                if em:
                    stats["per_field_accuracy"][card_type][fld][p_key]["exact"] += 1
                if v_func(pred_val):
                    stats["per_field_accuracy"][card_type][fld][p_key]["valid"] += 1
                stats["per_field_accuracy"][card_type][fld][p_key]["cer_sum"] += c_err

            if doc_exact:
                stats["pipelines"][p_key]["doc_exact_matches"] += 1

        sample_eval["fields_e"] = fields_e
        sample_eval["fields_d"] = fields_d
        sample_eval["fields_b"] = fields_b
        sample_eval["ground_truth"] = gt_fields
        stats["sample_details"].append(sample_eval)

        logger.info(
            f"  Kết quả: A={sum(1 for v in fields_a.values() if v)}/8, "
            f"B={sum(1 for v in fields_b.values() if v)}/8, "
            f"D={sum(1 for v in fields_d.values() if v)}/8, "
            f"E={sum(1 for v in fields_e.values() if v)}/8 "
            f"(Time D={lat_d:.1f}s, E={lat_e:.1f}s)"
        )

    # -------------------------------------------------------------
    # Tổng kết chỉ số thống kê cho từng Pipeline
    # -------------------------------------------------------------
    summary_table = []
    for p_key in pipeline_keys:
        p_data = stats["pipelines"][p_key]
        tot = p_data["total_fields"]
        extract_rate = round((p_data["extracted_fields"] / tot * 100) if tot > 0 else 0.0, 2)
        valid_rate = round((p_data["validated_fields"] / tot * 100) if tot > 0 else 0.0, 2)
        exact_rate = round((p_data["exact_matches"] / tot * 100) if tot > 0 else 0.0, 2)
        doc_acc = round((p_data["doc_exact_matches"] / len(sample_files) * 100) if sample_files else 0.0, 2)
        mean_cer = round(float(np.mean(p_data["cer_list"])) if p_data["cer_list"] else 0.0, 4)
        mean_wer = round(float(np.mean(p_data["wer_list"])) if p_data["wer_list"] else 0.0, 4)

        lats = p_data["latencies"]
        avg_lat = round(float(np.mean(lats)) if lats else 0.0, 2)
        med_lat = round(float(np.median(lats)) if lats else 0.0, 2)
        p95_lat = round(float(np.percentile(lats, 95)) if lats else 0.0, 2)

        p_data["metrics"] = {
            "field_extraction_rate_percent": extract_rate,
            "field_validation_rate_percent": valid_rate,
            "exact_match_accuracy_percent": exact_rate,
            "doc_level_accuracy_percent": doc_acc,
            "mean_cer": mean_cer,
            "mean_wer": mean_wer,
            "latency_avg_sec": avg_lat,
            "latency_median_sec": med_lat,
            "latency_p95_sec": p95_lat,
        }
        summary_table.append({
            "Pipeline": f"Pipeline {p_key} ({pipeline_names[p_key]})",
            "Extraction Rate": f"{extract_rate}%",
            "Validation Rate": f"{valid_rate}%",
            "Exact Match": f"{exact_rate}%",
            "CER": mean_cer,
            "WER": mean_wer,
            "Doc Accuracy": f"{doc_acc}%",
            "Avg Time": f"{avg_lat}s",
            "P95 Time": f"{p95_lat}s",
        })

    # Tính toán tỷ lệ hồi phục Fallback
    f_trig = stats["field_fallback_summary"]["total_triggered"]
    f_rec = stats["field_fallback_summary"]["total_recovered"]
    stats["field_fallback_summary"]["recovery_rate_percent"] = round((f_rec / f_trig * 100) if f_trig > 0 else 0.0, 2)

    report_path = out_dir / "benchmark_stage3_2_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("=========================================================================")
    logger.info("                     BENCHMARK STAGE 3.2 HOÀN TẤT                        ")
    logger.info("=========================================================================")
    for row in summary_table:
        logger.info(
            f"{row['Pipeline']:<48} | Ext={row['Extraction Rate']:<7} | Val={row['Validation Rate']:<7} | "
            f"EM={row['Exact Match']:<7} | CER={row['CER']:<6} | Time={row['Avg Time']}"
        )
    logger.info(f"Báo cáo chi tiết đã lưu tại: {report_path}")


if __name__ == "__main__":
    run_benchmark()

