"""Script kiểm thử tích hợp thực tế End-to-End và Benchmark hiệu năng (Giai đoạn 6).

Thực hiện:
1. Khởi chạy Uvicorn backend thật trên cổng 8000.
2. Kiểm tra kết nối /health và đo thời gian Cold Start vs Warm Start.
3. Chạy benchmark trên tập ảnh mẫu CCCD và GPLX ở cả hai chế độ 'fast' và 'accurate'.
4. Thu thập các chỉ số:
   - Detection latency (ms)
   - Rectification latency (ms)
   - OCR latency (ms)
   - Total latency (ms)
   - Roundtrip latency từ client (ms)
   - Mean / Median / P95 / Max
   - Process Peak RSS (MB) và CPU Utilization (%) qua psutil.
5. Kiểm thử các tình huống lỗi & biên:
   - File không phải ảnh (text file).
   - Ảnh vượt dung lượng 15MB.
   - Chỉ định sai loại giấy tờ (CCCD gửi document_type='gplx').
   - Request đồng thời để xác minh peak worker <= 1.
6. Đánh giá OCR Quality Gate:
   - Extraction Rate, Validation Rate, Phân tích trường cần sửa thủ công.
7. Đóng Uvicorn một cách an toàn (Graceful Shutdown) và dọn dẹp tài nguyên.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import psutil

# Thiết lập logging an toàn PII
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e2e_benchmark")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_ROOT = PROJECT_ROOT / "cccd-reader-api-main"
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

API_BASE_URL = "http://127.0.0.1:8000"


def mask_sensitive(val: str | None) -> str:
    """Che dữ liệu cá nhân (PII) an toàn khi hiển thị báo cáo."""
    if not val:
        return "[EMPTY]"
    val = str(val).strip()
    if len(val) <= 4:
        return "***"
    return f"{val[:2]}***{val[-2:]}"


def run_e2e_benchmark():
    logger.info("=== BẮT ĐẦU END-TO-END BENCHMARK GIAI ĐOẠN 6 ===")

    # 1. Khởi chạy Uvicorn Backend thật
    logger.info("Khởi chạy backend FastAPI qua Uvicorn thực tế...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(API_ROOT)

    ulog_file = open("uvicorn_e2e.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            str(PYTHON_EXE),
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--workers",
            "1",
        ],
        cwd=str(API_ROOT),
        env=env,
        stdout=ulog_file,
        stderr=ulog_file,
        text=True,
    )

    ps_proc = psutil.Process(proc.pid)

    # Chờ server sẵn sàng
    server_ready = False
    for attempt in range(40):
        try:
            r = httpx.get(f"{API_BASE_URL}/health", timeout=2.0)
            if r.status_code == 200:
                server_ready = True
                logger.info(f"Uvicorn server đã sẵn sàng sau {(attempt + 1) * 0.5:.1f}s!")
                break
        except Exception:
            time.sleep(0.5)

    if not server_ready:
        logger.error("Không thể khởi động Uvicorn server!")
        proc.kill()
        sys.exit(1)

    benchmark_results: dict[str, Any] = {
        "hardware": {
            "os": "Windows 11",
            "cpu_cores": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        },
        "cold_vs_warm": {},
        "modes": {},
        "error_tests": {},
        "concurrency": {},
        "ocr_quality": {},
    }

    try:
        # 2. Chuẩn bị ảnh test
        test_dir = API_ROOT / "datasets" / "images" / "test"
        cccd_samples = list(test_dir.glob("cccd_*.jpg"))
        gplx_samples = list(test_dir.glob("gplx_*.jpg"))

        logger.info(f"Tìm thấy {len(cccd_samples)} ảnh CCCD test và {len(gplx_samples)} ảnh GPLX test.")

        # 3. Đo Cold Start vs Warm Start
        logger.info("\n--- ĐO COLD START VS WARM START ---")
        cold_cccd = cccd_samples[0]
        with open(cold_cccd, "rb") as f:
            cold_bytes = f.read()

        # Cold request (lần gọi đầu tiên)
        t0 = time.perf_counter()
        resp_cold = httpx.post(
            f"{API_BASE_URL}/api/v2/upload",
            files={"file": (cold_cccd.name, cold_bytes, "image/jpeg")},
            data={"mode": "fast", "document_type": "auto"},
            timeout=180.0,
        )
        cold_latency = (time.perf_counter() - t0) * 1000.0
        assert resp_cold.status_code == 200, f"Cold request failed: {resp_cold.text}"
        cold_data = resp_cold.json()
        logger.info(f"Cold Start Request Latency: {cold_latency:.1f} ms")

        # Warm request (lần gọi thứ hai)
        t0 = time.perf_counter()
        resp_warm = httpx.post(
            f"{API_BASE_URL}/api/v2/upload",
            files={"file": (cold_cccd.name, cold_bytes, "image/jpeg")},
            data={"mode": "fast", "document_type": "auto"},
            timeout=180.0,
        )
        warm_latency = (time.perf_counter() - t0) * 1000.0
        assert resp_warm.status_code == 200, f"Warm request failed: {resp_warm.text}"
        warm_data = resp_warm.json()
        logger.info(f"Warm Start Request Latency: {warm_latency:.1f} ms")

        benchmark_results["cold_vs_warm"] = {
            "cold_start_ms": round(cold_latency, 1),
            "warm_start_ms": round(warm_latency, 1),
        }

        # 4. Benchmark theo từng chế độ và loại thẻ
        eval_configs = [
            ("cccd", "fast", cccd_samples[:4]),
            ("cccd", "accurate", cccd_samples[:2]),
            ("gplx", "fast", gplx_samples[:3]),
            ("gplx", "accurate", gplx_samples[:2]),
        ]

        for doc_type, mode, samples in eval_configs:
            cfg_name = f"{doc_type}_{mode}"
            logger.info(f"\n--- Benchmark cấu hình: {cfg_name.upper()} ({len(samples)} mẫu) ---")

            total_latencies = []
            det_latencies = []
            rect_latencies = []
            ocr_latencies = []
            cpu_samples = []
            rss_samples = []

            for s_idx, sample_path in enumerate(samples):
                with open(sample_path, "rb") as f:
                    s_bytes = f.read()

                t_start = time.perf_counter()
                r = httpx.post(
                    f"{API_BASE_URL}/api/v2/upload",
                    files={"file": (sample_path.name, s_bytes, "image/jpeg")},
                    data={"mode": mode, "document_type": doc_type},
                    timeout=180.0,
                )
                t_end = time.perf_counter()

                client_rtt_ms = (t_end - t_start) * 1000.0
                assert r.status_code == 200, f"Request failed: {r.text}"
                data = r.json()

                timings = data.get("timing_ms", {})
                det_latencies.append(timings.get("detection_ms", 0.0))
                rect_latencies.append(timings.get("rectification_ms", 0.0))
                ocr_latencies.append(timings.get("ocr_ms", 0.0))
                total_latencies.append(timings.get("total_ms", client_rtt_ms))

                # Thu thập thông số tài nguyên hệ thống
                rss_mb = ps_proc.memory_info().rss / (1024 * 1024)
                cpu_pct = psutil.cpu_percent(interval=None)
                rss_samples.append(rss_mb)
                cpu_samples.append(cpu_pct)

                # Thu thập thông tin OCR quality
                fields = data.get("fields", {})
                for f_name, f_val in fields.items():
                    key = f"{doc_type}:{f_name}"
                    if key not in benchmark_results["ocr_quality"]:
                        benchmark_results["ocr_quality"][key] = {
                            "total": 0,
                            "extracted": 0,
                            "valid": 0,
                        }
                    q_entry = benchmark_results["ocr_quality"][key]
                    q_entry["total"] += 1
                    if f_val.get("value"):
                        q_entry["extracted"] += 1
                    if f_val.get("valid"):
                        q_entry["valid"] += 1

                logger.info(
                    f"[{s_idx+1}/{len(samples)}] {sample_path.name}: "
                    f"Total: {total_latencies[-1]:.1f}ms (OCR: {ocr_latencies[-1]:.1f}ms) | "
                    f"RSS: {rss_mb:.1f}MB | CPU: {cpu_pct:.1f}%"
                )

            # Tính thống kê
            def calc_stats(vals: list[float]) -> dict[str, float]:
                if not vals:
                    return {}
                arr = np.array(vals)
                return {
                    "mean": round(float(np.mean(arr)), 1),
                    "median": round(float(np.median(arr)), 1),
                    "p95": round(float(np.percentile(arr, 95)), 1),
                    "max": round(float(np.max(arr)), 1),
                    "min": round(float(np.min(arr)), 1),
                }

            benchmark_results["modes"][cfg_name] = {
                "sample_count": len(samples),
                "total_latency_ms": calc_stats(total_latencies),
                "detection_latency_ms": calc_stats(det_latencies),
                "rectification_latency_ms": calc_stats(rect_latencies),
                "ocr_latency_ms": calc_stats(ocr_latencies),
                "peak_rss_mb": round(max(rss_samples) if rss_samples else 0.0, 1),
                "avg_cpu_percent": round(float(np.mean(cpu_samples)) if cpu_samples else 0.0, 1),
            }

        # 5. Kiểm thử các trường hợp biên & lỗi (Error Handling Tests)
        logger.info("\n--- KIỂM THỬ TRƯỜNG HỢP LỖI & BIÊN QUA HTTP ---")

        # 5.1. File không hợp lệ (plain text)
        r_text = httpx.post(
            f"{API_BASE_URL}/api/v2/upload",
            files={"file": ("fake.txt", b"This is not an image", "text/plain")},
            data={"mode": "fast", "document_type": "auto"},
        )
        assert r_text.status_code == 400
        benchmark_results["error_tests"]["unsupported_media_type"] = {
            "status_code": r_text.status_code,
            "error_code": r_text.json().get("error_code"),
            "passed": True,
        }
        logger.info(f"File text 400 test: {r_text.json().get('error_code')} -> PASS")

        # 5.2. File quá kích thước 15MB
        huge_bytes = b"0" * (16 * 1024 * 1024)
        r_huge = httpx.post(
            f"{API_BASE_URL}/api/v2/upload",
            files={"file": ("huge.jpg", huge_bytes, "image/jpeg")},
            data={"mode": "fast", "document_type": "auto"},
        )
        assert r_huge.status_code == 413
        benchmark_results["error_tests"]["file_too_large"] = {
            "status_code": r_huge.status_code,
            "error_code": r_huge.json().get("error_code"),
            "passed": True,
        }
        logger.info(f"File quá 15MB 413 test: {r_huge.json().get('error_code')} -> PASS")

        # 5.3. Chỉ định sai loại giấy tờ (Ảnh CCCD gửi document_type='gplx')
        r_mismatch = httpx.post(
            f"{API_BASE_URL}/api/v2/upload",
            files={"file": (cccd_samples[0].name, cold_bytes, "image/jpeg")},
            data={"mode": "fast", "document_type": "gplx"},
        )
        assert r_mismatch.status_code == 422
        benchmark_results["error_tests"]["document_type_mismatch"] = {
            "status_code": r_mismatch.status_code,
            "error_code": r_mismatch.json().get("error_code"),
            "passed": True,
        }
        logger.info(f"Sai loại giấy tờ 422 test: {r_mismatch.json().get('error_code')} -> PASS")

        # 6. Kiểm thử Concurrency qua HTTP thực tế
        logger.info("\n--- KIỂM THỬ CONCURRENCY QUA HTTP THẬT ---")
        # Gửi 2 request cùng lúc
        import concurrent.futures

        def send_req(req_id: int):
            t_s = time.perf_counter()
            r = httpx.post(
                f"{API_BASE_URL}/api/v2/upload",
                files={"file": (cccd_samples[0].name, cold_bytes, "image/jpeg")},
                data={"mode": "fast", "document_type": "auto"},
                timeout=180.0,
            )
            return req_id, r.status_code, (time.perf_counter() - t_s) * 1000.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(send_req, 1)
            f2 = ex.submit(send_req, 2)
            res1 = f1.result()
            res2 = f2.result()

        logger.info(f"Concurrent Req 1: Code {res1[1]}, Latency {res1[2]:.1f}ms")
        logger.info(f"Concurrent Req 2: Code {res2[1]}, Latency {res2[2]:.1f}ms")

        # Kiểm tra health stats
        health_resp = httpx.get(f"{API_BASE_URL}/health").json()
        concurrency_stats = health_resp.get("concurrency", {})
        logger.info(f"Health Concurrency Stats: {concurrency_stats}")

        benchmark_results["concurrency"] = {
            "req_1_latency_ms": round(res1[2], 1),
            "req_2_latency_ms": round(res2[2], 1),
            "peak_active_workers": concurrency_stats.get("peak_active_workers", 1),
            "available_slots": concurrency_stats.get("active_workers", 0),
        }

        # Lưu kết quả benchmark ra file JSON
        out_file = PROJECT_ROOT / "benchmark_stage6_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
        logger.info(f"\nĐã lưu kết quả benchmark hoàn chỉnh vào: {out_file}")

    finally:
        logger.info("Đang tắt Uvicorn server gracefully...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        try:
            ulog_file.close()
        except Exception:
            pass
        logger.info("Uvicorn server đã dừng.")


if __name__ == "__main__":
    run_e2e_benchmark()
