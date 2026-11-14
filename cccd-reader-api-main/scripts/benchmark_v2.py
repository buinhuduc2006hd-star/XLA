"""Benchmark độc lập đối sánh API v1, API v2 Fast, và API v2 Accurate (Giai đoạn 4 Hardening).

Đo lường các chỉ số chuẩn xác:
- Tách bạch phân loại CCCD và GPLX.
- Phân biệt mẫu có Ground Truth (tính Exact Match) và không có Ground Truth (Unseen).
- Thống kê độ trễ chi tiết: Mean, Median, P95, Min, Max (ms).
- Đo RAM peak thực tế trong suốt quá trình xử lý bằng background thread liên tục.
- Giám sát HTTP status code và tỷ lệ lỗi.
- BẢO MẬT TUYỆT ĐỐI (PII-Safe): Không ghi số định danh, họ tên, địa chỉ vào JSON hoặc Markdown.

Đầu ra:
- debug_output/stage4/benchmark_v2_report.json
- debug_output/stage4/benchmark_v2_summary.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
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


class PeakMemoryTracker:
    """Theo dõi lượng RAM tiêu thụ đỉnh (Peak RSS) liên tục trong suốt request."""

    def __init__(self, interval_sec: float = 0.02):
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.process = psutil.Process(os.getpid())
        self.peak_bytes = self.process.memory_info().rss

    def start(self) -> None:
        self.peak_bytes = self.process.memory_info().rss
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def _monitor(self) -> None:
        while not self._stop_event.is_set():
            try:
                mem = self.process.memory_info().rss
                if mem > self.peak_bytes:
                    self.peak_bytes = mem
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return round(self.peak_bytes / (1024 * 1024), 2)


def compute_latencies_stats(latencies: list[float]) -> dict[str, float]:
    """Tính toán Mean, Median, P95, Min, Max từ danh sách thời gian đo."""
    if not latencies:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(latencies)
    return {
        "mean": round(float(np.mean(arr)), 1),
        "median": round(float(np.median(arr)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
        "min": round(float(np.min(arr)), 1),
        "max": round(float(np.max(arr)), 1),
    }


def run_benchmark(
    sample_limit: int = 4,
    use_all_gt: bool = False,
    include_v1: bool = True,
):
    client = TestClient(app)
    output_dir = PROJECT_ROOT / "debug_output" / "stage4"
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_path = PROJECT_ROOT / "datasets" / "ground_truth_test_sample.json"
    ground_truth = {}
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

    test_images_dir = PROJECT_ROOT / "datasets" / "images" / "test"

    samples: list[dict[str, Any]] = []

    if use_all_gt:
        # Sử dụng toàn bộ các mẫu có Ground Truth
        idx = 1
        for filename, data in ground_truth.items():
            ctype = data.get("card_type", "cccd")
            samples.append({
                "sample_id": f"{ctype}_gt_{idx:02d}",
                "filename": filename,
                "type": ctype,
                "has_gt": True,
            })
            idx += 1
    else:
        # Tập mẫu tiêu chuẩn: 2 mẫu có Ground Truth (1 CCCD, 1 GPLX) và 2 mẫu Unseen
        samples = [
            {
                "sample_id": "cccd_gt_01",
                "filename": "cccd_1794065827572_499751437511627297_2033087232165695992_ffd2de2350e51f61744ef1ff9c530a1d_jpg.rf.5346da3778765ca825ef1a89d15591fc.jpg",
                "type": "cccd",
                "has_gt": True,
            },
            {
                "sample_id": "gplx_gt_01",
                "filename": "gplx_1794202637314_499751437511627297_2033087232165695992_5954b023295e96332ea71297c0c232af_jpg.rf.7133b3128f1cf3407b4c5b85da2f1838.jpg",
                "type": "gplx",
                "has_gt": True,
            },
            {
                "sample_id": "cccd_unseen_01",
                "filename": "cccd_1794065837682_499751437511627297_2033087232165695992_42827c9855282e14e1d4aa8d7cbe7d6b_jpg.rf.0bc3cfb8502154ecc1d38e0f2217e982.jpg",
                "type": "cccd",
                "has_gt": False,
            },
            {
                "sample_id": "gplx_unseen_01",
                "filename": "gplx_1794202640462_499751437511627297_2033087232165695992_3defb31a5d93179494de597c66899419_jpg.rf.6d440a42f77c8aef8759bfb81da0624a.jpg",
                "type": "gplx",
                "has_gt": False,
            },
        ]
        if sample_limit > 0 and sample_limit < len(samples):
            samples = samples[:sample_limit]

    pipeline_keys = ["v1", "v2_fast", "v2_accurate"] if include_v1 else ["v2_fast", "v2_accurate"]

    report: dict[str, Any] = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(samples),
        "gt_samples_count": sum(1 for s in samples if s["has_gt"]),
        "unseen_samples_count": sum(1 for s in samples if not s["has_gt"]),
        "pipelines": {
            k: {
                "name": (
                    "API v1 (Fixed ROI Baseline)" if k == "v1"
                    else "API v2 Fast (1-Pass)" if k == "v2_fast"
                    else "API v2 Accurate (Adaptive & Bounded Fallback)"
                ),
                "latencies_ms": [],
                "timing_breakdown": {
                    "detection_ms": [],
                    "rectification_ms": [],
                    "orientation_ms": [],
                    "ocr_ms": [],
                },
                "http_status_codes": {},
                "peak_ram_mb": 0.0,
                # Metrics chung
                "total_fields": 0,
                "extracted_fields": 0,
                "validated_fields": 0,
                "gt_reference_fields": 0,
                "exact_matches": 0,
                # Tách theo CCCD
                "cccd": {
                    "total_fields": 0,
                    "extracted_fields": 0,
                    "validated_fields": 0,
                    "gt_reference_fields": 0,
                    "exact_matches": 0,
                    "latencies_ms": [],
                },
                # Tách theo GPLX
                "gplx": {
                    "total_fields": 0,
                    "extracted_fields": 0,
                    "validated_fields": 0,
                    "gt_reference_fields": 0,
                    "exact_matches": 0,
                    "latencies_ms": [],
                },
            }
            for k in pipeline_keys
        },
        "sample_results": [],
    }

    mem_tracker = PeakMemoryTracker()
    logger.info(f"Khởi động Benchmark Stage 4 Hardening với {len(samples)} mẫu...")

    for sample in samples:
        img_path = test_images_dir / sample["filename"]
        if not img_path.exists():
            logger.warning(f"File {img_path} không tồn tại. Bỏ qua.")
            continue

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        card_type = sample["type"]
        has_gt = sample["has_gt"]
        gt_data = ground_truth.get(sample["filename"], {}) if has_gt else {}
        validators = CCCD_VALIDATORS if card_type == "cccd" else GPLX_VALIDATORS
        field_keys = list(validators.keys())
        expected_fields_count = len(field_keys)

        sample_record: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "card_type": card_type,
            "has_ground_truth": has_gt,
            "pipelines": {},
        }

        # -------------------------------------------------------------------
        # 1. Pipeline v1 Benchmark
        # -------------------------------------------------------------------
        if "v1" in pipeline_keys:
            logger.info(f"[{sample['sample_id']}] Chạy API v1...")
            mem_tracker.start()
            t0 = time.perf_counter()
            resp_v1 = client.post(
                "/api/v1/upload",
                files={"file": (sample["filename"], img_bytes, "image/jpeg")},
            )
            lat_v1_ms = (time.perf_counter() - t0) * 1000.0
            peak_v1 = mem_tracker.stop()

            st = resp_v1.status_code
            v1_data = resp_v1.json().get("data", {}) if st == 200 else {}

            v1_ext = 0
            v1_val = 0
            v1_exact = 0
            field_status_v1 = {}

            for fld in field_keys:
                raw_val = v1_data.get(fld)
                is_ext = raw_val is not None
                is_val = bool(is_ext and validators[fld](raw_val))
                is_ex = bool(has_gt and is_exact_match(raw_val, gt_data.get(fld)))

                if is_ext:
                    v1_ext += 1
                if is_val:
                    v1_val += 1
                if is_ex:
                    v1_exact += 1

                field_status_v1[fld] = {
                    "extracted": is_ext,
                    "valid": is_val,
                    "exact_match": is_ex if has_gt else None,
                }

            p_v1 = report["pipelines"]["v1"]
            p_v1["latencies_ms"].append(round(lat_v1_ms, 1))
            p_v1["http_status_codes"][st] = p_v1["http_status_codes"].get(st, 0) + 1
            p_v1["peak_ram_mb"] = max(p_v1["peak_ram_mb"], peak_v1)
            p_v1["total_fields"] += expected_fields_count
            p_v1["extracted_fields"] += v1_ext
            p_v1["validated_fields"] += v1_val
            if has_gt:
                p_v1["gt_reference_fields"] += expected_fields_count
                p_v1["exact_matches"] += v1_exact

            sub_v1 = p_v1[card_type]
            sub_v1["total_fields"] += expected_fields_count
            sub_v1["extracted_fields"] += v1_ext
            sub_v1["validated_fields"] += v1_val
            sub_v1["latencies_ms"].append(round(lat_v1_ms, 1))
            if has_gt:
                sub_v1["gt_reference_fields"] += expected_fields_count
                sub_v1["exact_matches"] += v1_exact

            sample_record["pipelines"]["v1"] = {
                "status_code": st,
                "latency_ms": round(lat_v1_ms, 1),
                "peak_ram_mb": peak_v1,
                "extracted": v1_ext,
                "validated": v1_val,
                "exact_matches": v1_exact if has_gt else None,
                "fields": field_status_v1,  # PII-Safe: chỉ lưu cờ boolean
            }

        # -------------------------------------------------------------------
        # 2. Pipeline v2 Fast Benchmark
        # -------------------------------------------------------------------
        if "v2_fast" in pipeline_keys:
            logger.info(f"[{sample['sample_id']}] Chạy API v2 Fast...")
            mem_tracker.start()
            t0 = time.perf_counter()
            resp_v2_f = client.post(
                "/api/v2/upload",
                files={"file": (sample["filename"], img_bytes, "image/jpeg")},
                data={"mode": "fast", "document_type": "auto"},
            )
            lat_v2_f_ms = (time.perf_counter() - t0) * 1000.0
            peak_v2_f = mem_tracker.stop()

            st = resp_v2_f.status_code
            v2_f_json = resp_v2_f.json() if st == 200 else {}
            v2_f_fields = v2_f_json.get("fields", {})

            f_ext = 0
            f_val = 0
            f_exact = 0
            field_status_f = {}

            for fld in field_keys:
                f_obj = v2_f_fields.get(fld, {})
                raw_val = f_obj.get("value")
                is_ext = raw_val is not None
                is_val = bool(f_obj.get("valid", False))
                is_ex = bool(has_gt and is_exact_match(raw_val, gt_data.get(fld)))

                if is_ext:
                    f_ext += 1
                if is_val:
                    f_val += 1
                if is_ex:
                    f_exact += 1

                field_status_f[fld] = {
                    "extracted": is_ext,
                    "valid": is_val,
                    "exact_match": is_ex if has_gt else None,
                    "source": f_obj.get("source"),
                }

            p_fast = report["pipelines"]["v2_fast"]
            p_fast["latencies_ms"].append(round(lat_v2_f_ms, 1))
            p_fast["http_status_codes"][st] = p_fast["http_status_codes"].get(st, 0) + 1
            p_fast["peak_ram_mb"] = max(p_fast["peak_ram_mb"], peak_v2_f)
            p_fast["total_fields"] += expected_fields_count
            p_fast["extracted_fields"] += f_ext
            p_fast["validated_fields"] += f_val
            if has_gt:
                p_fast["gt_reference_fields"] += expected_fields_count
                p_fast["exact_matches"] += f_exact

            timing_f = v2_f_json.get("timing_ms", {})
            for tk in ["detection_ms", "rectification_ms", "orientation_ms", "ocr_ms"]:
                if tk in timing_f:
                    p_fast["timing_breakdown"][tk].append(timing_f[tk])

            sub_f = p_fast[card_type]
            sub_f["total_fields"] += expected_fields_count
            sub_f["extracted_fields"] += f_ext
            sub_f["validated_fields"] += f_val
            sub_f["latencies_ms"].append(round(lat_v2_f_ms, 1))
            if has_gt:
                sub_f["gt_reference_fields"] += expected_fields_count
                sub_f["exact_matches"] += f_exact

            sample_record["pipelines"]["v2_fast"] = {
                "status_code": st,
                "latency_ms": round(lat_v2_f_ms, 1),
                "peak_ram_mb": peak_v2_f,
                "timing_breakdown": timing_f,
                "extracted": f_ext,
                "validated": f_val,
                "exact_matches": f_exact if has_gt else None,
                "fields": field_status_f,
            }

        # -------------------------------------------------------------------
        # 3. Pipeline v2 Accurate Benchmark
        # -------------------------------------------------------------------
        if "v2_accurate" in pipeline_keys:
            logger.info(f"[{sample['sample_id']}] Chạy API v2 Accurate...")
            mem_tracker.start()
            t0 = time.perf_counter()
            resp_v2_a = client.post(
                "/api/v2/upload",
                files={"file": (sample["filename"], img_bytes, "image/jpeg")},
                data={"mode": "accurate", "document_type": "auto"},
            )
            lat_v2_a_ms = (time.perf_counter() - t0) * 1000.0
            peak_v2_a = mem_tracker.stop()

            st = resp_v2_a.status_code
            v2_a_json = resp_v2_a.json() if st == 200 else {}
            v2_a_fields = v2_a_json.get("fields", {})

            a_ext = 0
            a_val = 0
            a_exact = 0
            field_status_a = {}

            for fld in field_keys:
                a_obj = v2_a_fields.get(fld, {})
                raw_val = a_obj.get("value")
                is_ext = raw_val is not None
                is_val = bool(a_obj.get("valid", False))
                is_ex = bool(has_gt and is_exact_match(raw_val, gt_data.get(fld)))

                if is_ext:
                    a_ext += 1
                if is_val:
                    a_val += 1
                if is_ex:
                    a_exact += 1

                field_status_a[fld] = {
                    "extracted": is_ext,
                    "valid": is_val,
                    "exact_match": is_ex if has_gt else None,
                    "source": a_obj.get("source"),
                }

            p_acc = report["pipelines"]["v2_accurate"]
            p_acc["latencies_ms"].append(round(lat_v2_a_ms, 1))
            p_acc["http_status_codes"][st] = p_acc["http_status_codes"].get(st, 0) + 1
            p_acc["peak_ram_mb"] = max(p_acc["peak_ram_mb"], peak_v2_a)
            p_acc["total_fields"] += expected_fields_count
            p_acc["extracted_fields"] += a_ext
            p_acc["validated_fields"] += a_val
            if has_gt:
                p_acc["gt_reference_fields"] += expected_fields_count
                p_acc["exact_matches"] += a_exact

            timing_a = v2_a_json.get("timing_ms", {})
            for tk in ["detection_ms", "rectification_ms", "orientation_ms", "ocr_ms"]:
                if tk in timing_a:
                    p_acc["timing_breakdown"][tk].append(timing_a[tk])

            sub_a = p_acc[card_type]
            sub_a["total_fields"] += expected_fields_count
            sub_a["extracted_fields"] += a_ext
            sub_a["validated_fields"] += a_val
            sub_a["latencies_ms"].append(round(lat_v2_a_ms, 1))
            if has_gt:
                sub_a["gt_reference_fields"] += expected_fields_count
                sub_a["exact_matches"] += a_exact

            sample_record["pipelines"]["v2_accurate"] = {
                "status_code": st,
                "latency_ms": round(lat_v2_a_ms, 1),
                "peak_ram_mb": peak_v2_a,
                "timing_breakdown": timing_a,
                "extracted": a_ext,
                "validated": a_val,
                "exact_matches": a_exact if has_gt else None,
                "fields": field_status_a,
            }

        report["sample_results"].append(sample_record)

    # -----------------------------------------------------------------------
    # 4. Tính toán Metrics tổng hợp và Thống kê phân vị
    # -----------------------------------------------------------------------
    for p_key, p_data in report["pipelines"].items():
        tot = max(1, p_data["total_fields"])
        gt_tot = max(1, p_data["gt_reference_fields"])

        p_data["overall_extraction_rate_percent"] = round((p_data["extracted_fields"] / tot) * 100, 2)
        p_data["overall_validation_rate_percent"] = round((p_data["validated_fields"] / tot) * 100, 2)
        p_data["overall_exact_match_percent"] = (
            round((p_data["exact_matches"] / gt_tot) * 100, 2) if p_data["gt_reference_fields"] > 0 else None
        )

        p_data["latency_stats_ms"] = compute_latencies_stats(p_data["latencies_ms"])

        for ctype in ["cccd", "gplx"]:
            sub = p_data[ctype]
            s_tot = max(1, sub["total_fields"])
            s_gt = max(1, sub["gt_reference_fields"])
            sub["extraction_rate_percent"] = round((sub["extracted_fields"] / s_tot) * 100, 2)
            sub["validation_rate_percent"] = round((sub["validated_fields"] / s_tot) * 100, 2)
            sub["exact_match_percent"] = (
                round((sub["exact_matches"] / s_gt) * 100, 2) if sub["gt_reference_fields"] > 0 else None
            )
            sub["latency_stats_ms"] = compute_latencies_stats(sub["latencies_ms"])

        # Breakdown stats
        p_data["timing_breakdown_stats_ms"] = {
            tk: compute_latencies_stats(p_data["timing_breakdown"][tk])
            for tk in ["detection_ms", "rectification_ms", "orientation_ms", "ocr_ms"]
        }

    # Xuất file JSON
    out_json = output_dir / "benchmark_v2_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Báo cáo JSON đã lưu tại: {out_json}")

    # Xuất file Markdown Summary
    out_md = output_dir / "benchmark_v2_summary.md"
    _generate_markdown_summary(report, out_md)
    logger.info(f"Báo cáo Markdown đã lưu tại: {out_md}")

    # In kết quả console
    _print_console_summary(report)


def _generate_markdown_summary(report: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# BÁO CÁO KẾT QUẢ BENCHMARK GIAI ĐOẠN 4 HARDENING",
        f"- **Thời điểm đánh giá:** {report['benchmark_timestamp']}",
        f"- **Tổng số mẫu:** {report['total_samples']} (Ground Truth: {report['gt_samples_count']}, Unseen: {report['unseen_samples_count']})",
        "",
        "## 1. Bảng đối sánh tổng thể (Overall Comparison)",
        "",
        "| Chỉ số | API v1 (Baseline) | API v2 Fast | API v2 Accurate |",
        "|---|---:|---:|---:|",
    ]

    p = report["pipelines"]
    v1 = p.get("v1", {})
    f = p.get("v2_fast", {})
    a = p.get("v2_accurate", {})

    def _fmt_pct(val):
        return f"{val:.1f}%" if val is not None else "N/A"

    def _fmt_sec(val_ms):
        return f"{val_ms / 1000.0:.2f} s" if val_ms else "0.00 s"

    lines.append(f"| **Field Extraction Rate** | {_fmt_pct(v1.get('overall_extraction_rate_percent'))} | {_fmt_pct(f.get('overall_extraction_rate_percent'))} | {_fmt_pct(a.get('overall_extraction_rate_percent'))} |")
    lines.append(f"| **Field Validation Rate** | {_fmt_pct(v1.get('overall_validation_rate_percent'))} | {_fmt_pct(f.get('overall_validation_rate_percent'))} | {_fmt_pct(a.get('overall_validation_rate_percent'))} |")
    lines.append(f"| **Exact Match (trên GT)** | {_fmt_pct(v1.get('overall_exact_match_percent'))} | {_fmt_pct(f.get('overall_exact_match_percent'))} | {_fmt_pct(a.get('overall_exact_match_percent'))} |")
    lines.append(f"| **Độ trễ trung bình (Mean)** | {_fmt_sec(v1.get('latency_stats_ms', {}).get('mean', 0))} | {_fmt_sec(f.get('latency_stats_ms', {}).get('mean', 0))} | {_fmt_sec(a.get('latency_stats_ms', {}).get('mean', 0))} |")
    lines.append(f"| **Độ trễ trung vị (Median)** | {_fmt_sec(v1.get('latency_stats_ms', {}).get('median', 0))} | {_fmt_sec(f.get('latency_stats_ms', {}).get('median', 0))} | {_fmt_sec(a.get('latency_stats_ms', {}).get('median', 0))} |")
    lines.append(f"| **Độ trễ P95** | {_fmt_sec(v1.get('latency_stats_ms', {}).get('p95', 0))} | {_fmt_sec(f.get('latency_stats_ms', {}).get('p95', 0))} | {_fmt_sec(a.get('latency_stats_ms', {}).get('p95', 0))} |")
    lines.append(f"| **RAM tiêu thụ đỉnh (Peak)** | {v1.get('peak_ram_mb', 0)} MB | {f.get('peak_ram_mb', 0)} MB | {a.get('peak_ram_mb', 0)} MB |")

    lines.extend([
        "",
        "## 2. Phân tích chi tiết theo loại giấy tờ (CCCD vs GPLX)",
        "",
        "### Thẻ Căn cước công dân (CCCD - 8 trường)",
        "| Pipeline | Extraction Rate | Validation Rate | Exact Match | Latency Mean |",
        "|---|---:|---:|---:|---:|",
    ])
    for pk, name in [("v1", "API v1"), ("v2_fast", "API v2 Fast"), ("v2_accurate", "API v2 Accurate")]:
        sub = p.get(pk, {}).get("cccd", {})
        lines.append(f"| {name} | {_fmt_pct(sub.get('extraction_rate_percent'))} | {_fmt_pct(sub.get('validation_rate_percent'))} | {_fmt_pct(sub.get('exact_match_percent'))} | {_fmt_sec(sub.get('latency_stats_ms', {}).get('mean', 0))} |")

    lines.extend([
        "",
        "### Giấy phép lái xe (GPLX - 7 trường)",
        "| Pipeline | Extraction Rate | Validation Rate | Exact Match | Latency Mean |",
        "|---|---:|---:|---:|---:|",
    ])
    for pk, name in [("v1", "API v1"), ("v2_fast", "API v2 Fast"), ("v2_accurate", "API v2 Accurate")]:
        sub = p.get(pk, {}).get("gplx", {})
        lines.append(f"| {name} | {_fmt_pct(sub.get('extraction_rate_percent'))} | {_fmt_pct(sub.get('validation_rate_percent'))} | {_fmt_pct(sub.get('exact_match_percent'))} | {_fmt_sec(sub.get('latency_stats_ms', {}).get('mean', 0))} |")

    lines.extend([
        "",
        "## 3. Phân rã độ trễ Pipeline V2 (Timing Breakdown - Mean)",
        "",
        "| Giai đoạn | API v2 Fast | API v2 Accurate |",
        "|---|---:|---:|",
    ])
    for tk, lbl in [
        ("detection_ms", "1. YOLOv8n Document Detection"),
        ("rectification_ms", "2. OpenCV Document Rectification"),
        ("orientation_ms", "3. Orientation Correction"),
        ("ocr_ms", "4. Adaptive OCR & Validation"),
    ]:
        f_m = f.get("timing_breakdown_stats_ms", {}).get(tk, {}).get("mean", 0)
        a_m = a.get("timing_breakdown_stats_ms", {}).get(tk, {}).get("mean", 0)
        lines.append(f"| {lbl} | {f_m:.1f} ms | {a_m:.1f} ms |")

    lines.extend([
        "",
        "## 4. Bảo mật dữ liệu & Quyền riêng tư (PII Compliance)",
        "- Báo cáo benchmark này và file JSON tương ứng **không lưu trữ** bất kỳ nội dung văn bản cá nhân nào (số CCCD, họ tên, địa chỉ, ngày sinh).",
        "- Tất cả dữ liệu đánh giá chỉ sử dụng mã mẫu định danh phân loại (`sample_id`) và cờ boolean xác thực.",
    ])

    with open(out_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(lines) + "\n")


def _print_console_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 76)
    print("KẾT QUẢ BENCHMARK GIAI ĐOẠN 4 HARDENING (V1 VS V2 FAST VS V2 ACCURATE)")
    print("=" * 76)
    for pk, pdata in report["pipelines"].items():
        print(f"Pipeline: {pdata['name']}")
        print(f"  - Extraction Rate:  {pdata['overall_extraction_rate_percent']}% ({pdata['extracted_fields']}/{pdata['total_fields']})")
        print(f"  - Validation Rate:  {pdata['overall_validation_rate_percent']}% ({pdata['validated_fields']}/{pdata['total_fields']})")
        em = pdata['overall_exact_match_percent']
        em_str = f"{em}% ({pdata['exact_matches']}/{pdata['gt_reference_fields']})" if em is not None else "N/A"
        print(f"  - Exact Match (GT): {em_str}")
        lstats = pdata["latency_stats_ms"]
        print(f"  - Latency:          Mean={lstats['mean']/1000.0:.2f}s | Median={lstats['median']/1000.0:.2f}s | P95={lstats['p95']/1000.0:.2f}s | Min={lstats['min']/1000.0:.2f}s | Max={lstats['max']/1000.0:.2f}s")
        print(f"  - Peak RAM:         {pdata['peak_ram_mb']} MB")
        print(f"  - HTTP Status:      {pdata['http_status_codes']}")
        print("-" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Benchmark V2 Hardening")
    parser.add_argument("--samples", type=int, default=4, help="Số lượng mẫu test (mặc định 4)")
    parser.add_argument("--all-gt", action="store_true", help="Chạy trên tất cả 6 mẫu có Ground Truth")
    parser.add_argument("--no-v1", action="store_true", help="Bỏ qua API v1 để tăng tốc độ chạy")
    args = parser.parse_args()

    run_benchmark(
        sample_limit=args.samples,
        use_all_gt=args.all_gt,
        include_v1=not args.no_v1,
    )
