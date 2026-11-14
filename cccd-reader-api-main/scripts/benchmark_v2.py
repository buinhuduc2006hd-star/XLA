"""Benchmark độc lập đối sánh API v1, API v2 Fast, và API v2 Accurate (Giai đoạn 4).

Đo lường các chỉ số:
- HTTP Status Code
- Detection Confidence & Bounding Box
- Rectification Success & Method
- Orientation Correction & Angles
- Field Extraction Rate (%)
- Field Validation Rate (%)
- Exact-Match Accuracy (%) (trên tập Ground Truth)
- End-to-end Latency (ms) và Breakdown (detection, rectification, orientation, ocr)
- Peak RAM (MB) bằng psutil

Đầu ra:
- debug_output/stage4/benchmark_v2_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import psutil
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import app
from schemas import DocumentV2Response
from services.adaptive_extractor import CCCD_VALIDATORS, GPLX_VALIDATORS, is_exact_match

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_current_process_ram_mb() -> float:
    """Đo lượng RAM tiêu thụ hiện tại của process tính bằng MB."""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def run_benchmark():
    client = TestClient(app)
    output_dir = PROJECT_ROOT / "debug_output" / "stage4"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Đọc Ground Truth nếu có
    gt_path = PROJECT_ROOT / "datasets" / "ground_truth_test_sample.json"
    ground_truth = {}
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

    test_images_dir = PROJECT_ROOT / "datasets" / "images" / "test"

    # Chọn 2 ảnh mẫu đối chứng có Ground Truth (1 CCCD, 1 GPLX)
    # và 2 ảnh mẫu kiểm chứng mới (Unseen) để đánh giá tính tổng quát hóa
    samples = [
        {
            "name": "cccd_gt_sample",
            "filename": "cccd_1794065827572_499751437511627297_2033087232165695992_ffd2de2350e51f61744ef1ff9c530a1d_jpg.rf.5346da3778765ca825ef1a89d15591fc.jpg",
            "type": "cccd",
            "is_unseen": False,
        },
        {
            "name": "gplx_gt_sample",
            "filename": "gplx_1794202637314_499751437511627297_2033087232165695992_5954b023295e96332ea71297c0c232af_jpg.rf.7133b3128f1cf3407b4c5b85da2f1838.jpg",
            "type": "gplx",
            "is_unseen": False,
        },
        {
            "name": "cccd_unseen_sample",
            "filename": "cccd_1794065837682_499751437511627297_2033087232165695992_42827c9855282e14e1d4aa8d7cbe7d6b_jpg.rf.0bc3cfb8502154ecc1d38e0f2217e982.jpg",
            "type": "cccd",
            "is_unseen": True,
        },
        {
            "name": "gplx_unseen_sample",
            "filename": "gplx_1794202640462_499751437511627297_2033087232165695992_3defb31a5d93179494de597c66899419_jpg.rf.6d440a42f77c8aef8759bfb81da0624a.jpg",
            "type": "gplx",
            "is_unseen": True,
        },
    ]

    report = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipelines": {
            "v1": {
                "name": "API v1 (/api/v1/upload - Fixed ROI Baseline)",
                "latencies_ms": [],
                "extracted_fields": 0,
                "validated_fields": 0,
                "exact_matches": 0,
                "total_fields": 0,
                "peak_ram_mb": 0.0,
            },
            "v2_fast": {
                "name": "API v2 Fast (/api/v2/upload - Mode Fast)",
                "latencies_ms": [],
                "timing_breakdown": {
                    "detection_ms": [],
                    "rectification_ms": [],
                    "orientation_ms": [],
                    "ocr_ms": [],
                },
                "extracted_fields": 0,
                "validated_fields": 0,
                "exact_matches": 0,
                "total_fields": 0,
                "peak_ram_mb": 0.0,
            },
            "v2_accurate": {
                "name": "API v2 Accurate (/api/v2/upload - Mode Accurate)",
                "latencies_ms": [],
                "timing_breakdown": {
                    "detection_ms": [],
                    "rectification_ms": [],
                    "orientation_ms": [],
                    "ocr_ms": [],
                },
                "extracted_fields": 0,
                "validated_fields": 0,
                "exact_matches": 0,
                "total_fields": 0,
                "peak_ram_mb": 0.0,
            },
        },
        "sample_results": [],
    }

    initial_ram = get_current_process_ram_mb()
    max_ram = initial_ram
    logger.info(f"Khởi động Benchmark V1 vs V2. Initial RAM: {initial_ram} MB")

    for sample in samples:
        img_path = test_images_dir / sample["filename"]
        if not img_path.exists():
            logger.warning(f"Không tìm thấy file ảnh: {img_path}. Bỏ qua.")
            continue

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        gt_fields = ground_truth.get(sample["filename"], {})
        expected_type = sample["type"]
        validators = CCCD_VALIDATORS if expected_type == "cccd" else GPLX_VALIDATORS
        field_keys = list(validators.keys())

        sample_res = {
            "name": sample["name"],
            "filename": sample["filename"],
            "is_unseen": sample["is_unseen"],
            "expected_type": expected_type,
            "v1": {},
            "v2_fast": {},
            "v2_accurate": {},
        }

        # -------------------------------------------------------------------
        # 1. API v1 Benchmark
        # -------------------------------------------------------------------
        logger.info(f"[{sample['name']}] Đang chạy API v1...")
        t0 = time.perf_counter()
        resp_v1 = client.post(
            "/api/v1/upload",
            files={"file": (sample["filename"], img_bytes, "image/jpeg")},
        )
        t_v1_ms = (time.perf_counter() - t0) * 1000.0
        cur_ram = get_current_process_ram_mb()
        max_ram = max(max_ram, cur_ram)

        v1_data = resp_v1.json().get("data", {}) if resp_v1.status_code == 200 else {}
        v1_ext = sum(1 for k in field_keys if v1_data.get(k) is not None)
        v1_val = sum(1 for k in field_keys if validators[k](v1_data.get(k)))
        v1_exact = 0
        if gt_fields:
            v1_exact = sum(1 for k in field_keys if is_exact_match(v1_data.get(k), gt_fields.get(k)))

        report["pipelines"]["v1"]["latencies_ms"].append(round(t_v1_ms, 1))
        report["pipelines"]["v1"]["extracted_fields"] += v1_ext
        report["pipelines"]["v1"]["validated_fields"] += v1_val
        report["pipelines"]["v1"]["exact_matches"] += v1_exact
        report["pipelines"]["v1"]["total_fields"] += len(field_keys)

        sample_res["v1"] = {
            "status_code": resp_v1.status_code,
            "latency_ms": round(t_v1_ms, 1),
            "extracted": v1_ext,
            "validated": v1_val,
            "exact_matches": v1_exact if gt_fields else None,
            "fields": v1_data,
        }

        # -------------------------------------------------------------------
        # 2. API v2 Fast Benchmark
        # -------------------------------------------------------------------
        logger.info(f"[{sample['name']}] Đang chạy API v2 (Fast)...")
        t0 = time.perf_counter()
        resp_v2_f = client.post(
            "/api/v2/upload",
            files={"file": (sample["filename"], img_bytes, "image/jpeg")},
            data={"mode": "fast", "document_type": "auto"},
        )
        t_v2_f_ms = (time.perf_counter() - t0) * 1000.0
        cur_ram = get_current_process_ram_mb()
        max_ram = max(max_ram, cur_ram)

        v2_f_json = resp_v2_f.json() if resp_v2_f.status_code == 200 else {}
        v2_f_fields = v2_f_json.get("fields", {})
        v2_f_ext = sum(1 for k in field_keys if v2_f_fields.get(k, {}).get("value") is not None)
        v2_f_val = sum(1 for k in field_keys if v2_f_fields.get(k, {}).get("valid") is True)
        v2_f_exact = 0
        if gt_fields:
            v2_f_exact = sum(1 for k in field_keys if is_exact_match(v2_f_fields.get(k, {}).get("value"), gt_fields.get(k)))

        report["pipelines"]["v2_fast"]["latencies_ms"].append(round(t_v2_f_ms, 1))
        report["pipelines"]["v2_fast"]["extracted_fields"] += v2_f_ext
        report["pipelines"]["v2_fast"]["validated_fields"] += v2_f_val
        report["pipelines"]["v2_fast"]["exact_matches"] += v2_f_exact
        report["pipelines"]["v2_fast"]["total_fields"] += len(field_keys)

        timing_f = v2_f_json.get("timing_ms", {})
        for tk in ["detection_ms", "rectification_ms", "orientation_ms", "ocr_ms"]:
            if tk in timing_f:
                report["pipelines"]["v2_fast"]["timing_breakdown"][tk].append(timing_f[tk])

        sample_res["v2_fast"] = {
            "status_code": resp_v2_f.status_code,
            "latency_ms": round(t_v2_f_ms, 1),
            "timing_breakdown": timing_f,
            "detection": v2_f_json.get("detection"),
            "processing": v2_f_json.get("processing"),
            "extracted": v2_f_ext,
            "validated": v2_f_val,
            "exact_matches": v2_f_exact if gt_fields else None,
            "fields": {k: v.get("value") for k, v in v2_f_fields.items()},
        }

        # -------------------------------------------------------------------
        # 3. API v2 Accurate Benchmark
        # -------------------------------------------------------------------
        logger.info(f"[{sample['name']}] Đang chạy API v2 (Accurate)...")
        t0 = time.perf_counter()
        resp_v2_a = client.post(
            "/api/v2/upload",
            files={"file": (sample["filename"], img_bytes, "image/jpeg")},
            data={"mode": "accurate", "document_type": "auto"},
        )
        t_v2_a_ms = (time.perf_counter() - t0) * 1000.0
        cur_ram = get_current_process_ram_mb()
        max_ram = max(max_ram, cur_ram)

        v2_a_json = resp_v2_a.json() if resp_v2_a.status_code == 200 else {}
        v2_a_fields = v2_a_json.get("fields", {})
        v2_a_ext = sum(1 for k in field_keys if v2_a_fields.get(k, {}).get("value") is not None)
        v2_a_val = sum(1 for k in field_keys if v2_a_fields.get(k, {}).get("valid") is True)
        v2_a_exact = 0
        if gt_fields:
            v2_a_exact = sum(1 for k in field_keys if is_exact_match(v2_a_fields.get(k, {}).get("value"), gt_fields.get(k)))

        report["pipelines"]["v2_accurate"]["latencies_ms"].append(round(t_v2_a_ms, 1))
        report["pipelines"]["v2_accurate"]["extracted_fields"] += v2_a_ext
        report["pipelines"]["v2_accurate"]["validated_fields"] += v2_a_val
        report["pipelines"]["v2_accurate"]["exact_matches"] += v2_a_exact
        report["pipelines"]["v2_accurate"]["total_fields"] += len(field_keys)

        timing_a = v2_a_json.get("timing_ms", {})
        for tk in ["detection_ms", "rectification_ms", "orientation_ms", "ocr_ms"]:
            if tk in timing_a:
                report["pipelines"]["v2_accurate"]["timing_breakdown"][tk].append(timing_a[tk])

        sample_res["v2_accurate"] = {
            "status_code": resp_v2_a.status_code,
            "latency_ms": round(t_v2_a_ms, 1),
            "timing_breakdown": timing_a,
            "detection": v2_a_json.get("detection"),
            "processing": v2_a_json.get("processing"),
            "extracted": v2_a_ext,
            "validated": v2_a_val,
            "exact_matches": v2_a_exact if gt_fields else None,
            "fields": {k: v.get("value") for k, v in v2_a_fields.items()},
        }

        report["sample_results"].append(sample_res)

    # Tính toán metrics tổng hợp
    for p_key, p_data in report["pipelines"].items():
        total = max(1, p_data["total_fields"])
        p_data["extraction_rate_percent"] = round((p_data["extracted_fields"] / total) * 100, 2)
        p_data["validation_rate_percent"] = round((p_data["validated_fields"] / total) * 100, 2)
        # Chỉ tính exact match trên 15 trường có ground truth (8 cccd + 7 gplx = 15)
        p_data["exact_match_rate_percent"] = round((p_data["exact_matches"] / 15) * 100, 2)
        lats = p_data["latencies_ms"]
        p_data["latency_avg_ms"] = round(sum(lats) / len(lats), 1) if lats else 0.0
        p_data["latency_avg_sec"] = round((p_data["latency_avg_ms"] / 1000.0), 2)
        p_data["peak_ram_mb"] = round(max_ram, 1)

    out_file = output_dir / "benchmark_v2_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Hoàn thành Benchmark V2! Báo cáo đã lưu tại: {out_file}")
    print("\n" + "=" * 70)
    print("KẾT QUẢ BENCHMARK GIAI ĐOẠN 4 (API v1 vs API v2 Fast vs API v2 Accurate)")
    print("=" * 70)
    for k, v in report["pipelines"].items():
        print(f"Pipeline: {v['name']}")
        print(f"  - Extraction Rate:  {v['extraction_rate_percent']}% ({v['extracted_fields']}/{v['total_fields']})")
        print(f"  - Validation Rate:  {v['validation_rate_percent']}% ({v['validated_fields']}/{v['total_fields']})")
        print(f"  - Exact Match (GT): {v['exact_match_rate_percent']}% ({v['exact_matches']}/15)")
        print(f"  - Latency Trung bình: {v['latency_avg_sec']} s / ảnh ({v['latency_avg_ms']} ms)")
        print(f"  - Peak RAM:         {v['peak_ram_mb']} MB")
        print("-" * 70)


if __name__ == "__main__":
    run_benchmark()

