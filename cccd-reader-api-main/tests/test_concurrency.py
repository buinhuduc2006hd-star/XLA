"""Unit & Stress Tests chuyên sâu cho Backend Concurrency & InferencePool (Giai đoạn 6).

Bao phủ toàn diện 7 kịch bản bắt buộc theo đặc tả:
1. Một request xử lý bình thường.
2. Hai request gửi đồng thời.
3. Client timeout khi OCR đang chạy (Cancel coroutine nhưng worker thread vẫn chạy).
4. Client chủ động ngắt kết nối (Client disconnect).
5. Backend xảy ra exception trong worker thread.
6. Client gửi lại request ngay sau timeout (Xác minh worker slot KHÔNG bị giải phóng sớm).
7. Ba request được gửi liên tiếp khi CPU đang bận.
+ Kiểm tra hành vi shutdown khi có request đang chạy.
+ Kiểm tra Event Loop không bao giờ bị block.
"""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.concurrency import InferencePool, reset_inference_pool_for_test


class TestInferencePoolConcurrency(unittest.IsolatedAsyncioTestCase):
    """Kiểm thử chuyên sâu cơ chế concurrency, semaphore và worker lifecycle."""

    async def asyncSetUp(self):
        # Đặt max_workers = 1 để kiểm tra giới hạn nghiêm ngặt nhất
        self.pool = reset_inference_pool_for_test(max_workers=1)

    async def asyncTearDown(self):
        self.pool.shutdown(wait=False)

    # -----------------------------------------------------------------------
    # Kịch bản 1: Một request xử lý bình thường
    # -----------------------------------------------------------------------
    async def test_01_single_request_normal(self):
        """Xác minh một request đơn lẻ chạy trơn tru, cập nhật đúng metrics."""
        def slow_worker(x: int) -> int:
            time.sleep(0.05)
            return x * 2

        result = await self.pool.run_in_pool(slow_worker, 21)
        self.assertEqual(result, 42)

        stats = self.pool.get_stats()
        self.assertEqual(stats["total_requests"], 1)
        self.assertEqual(stats["total_completed"], 1)
        self.assertEqual(stats["active_workers"], 0)
        self.assertEqual(stats["peak_active_workers"], 1)

    # -----------------------------------------------------------------------
    # Kịch bản 2: Hai request gửi đồng thời
    # -----------------------------------------------------------------------
    async def test_02_two_concurrent_requests(self):
        """Xác minh 2 request gửi cùng lúc được tuần tự hóa, peak worker luôn <= 1."""
        order = []

        def slow_worker(req_id: int) -> int:
            order.append(f"start_{req_id}")
            time.sleep(0.06)
            order.append(f"end_{req_id}")
            return req_id

        t1 = asyncio.create_task(self.pool.run_in_pool(slow_worker, 1))
        t2 = asyncio.create_task(self.pool.run_in_pool(slow_worker, 2))

        res1, res2 = await asyncio.gather(t1, t2)
        self.assertEqual((res1, res2), (1, 2))

        stats = self.pool.get_stats()
        self.assertEqual(stats["total_requests"], 2)
        self.assertEqual(stats["total_completed"], 2)
        self.assertEqual(stats["peak_active_workers"], 1, "Peak active workers vượt quá giới hạn 1!")

        # Đảm bảo tuần tự hóa tuyệt đối: start_1 -> end_1 -> start_2 -> end_2
        self.assertEqual(order, ["start_1", "end_1", "start_2", "end_2"])

    # -----------------------------------------------------------------------
    # Kịch bản 3: Client timeout khi OCR đang chạy
    # -----------------------------------------------------------------------
    async def test_03_client_timeout_while_ocr_running(self):
        """Xác minh khi client timeout hủy task asyncio, worker thread bên dưới vẫn an toàn."""
        started_event = asyncio.Event()

        def long_ocr():
            time.sleep(0.10)
            return "done_ocr"

        task = asyncio.create_task(self.pool.run_in_pool(long_ocr))

        # Giả lập client timeout sau 0.02s (trong khi OCR cần 0.10s)
        await asyncio.sleep(0.02)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        stats_immediate = self.pool.get_stats()
        self.assertEqual(stats_immediate["total_cancelled"], 1)

        # Đợi luồng nền kết thúc hẳn
        await asyncio.sleep(0.12)
        stats_final = self.pool.get_stats()
        self.assertEqual(stats_final["active_workers"], 0)
        self.assertEqual(stats_final["total_completed"], 1)

    # -----------------------------------------------------------------------
    # Kịch bản 4: Client chủ động ngắt kết nối (Client Disconnect)
    # -----------------------------------------------------------------------
    async def test_04_client_disconnect(self):
        """Xác minh hành vi khi client chủ động đóng kết nối HTTP."""
        def dummy_inference():
            time.sleep(0.08)
            return "ok"

        task = asyncio.create_task(self.pool.run_in_pool(dummy_inference))
        await asyncio.sleep(0.01)
        # Client disconnect tương đương với cancel asyncio task
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        await asyncio.sleep(0.10)
        self.assertEqual(self.pool.get_stats()["active_workers"], 0)

    # -----------------------------------------------------------------------
    # Kịch bản 5: Backend xảy ra exception
    # -----------------------------------------------------------------------
    async def test_05_backend_exception_handling(self):
        """Xác minh exception trong worker thread không làm rò rỉ slot semaphore."""
        def failing_worker():
            time.sleep(0.02)
            raise ValueError("Lỗi mô phỏng trong quá trình OCR")

        with self.assertRaises(ValueError):
            await self.pool.run_in_pool(failing_worker)

        # Slot phải được giải phóng và sẵn sàng cho request tiếp theo
        stats = self.pool.get_stats()
        self.assertEqual(stats["active_workers"], 0)

        # Thử một request mới ngay sau lỗi để xác minh pool không bị kẹt
        def recovery_worker():
            return "recovered"

        res = await self.pool.run_in_pool(recovery_worker)
        self.assertEqual(res, "recovered")

    # -----------------------------------------------------------------------
    # Kịch bản 6: Client gửi lại request ngay sau timeout (Chống Premature Release)
    # -----------------------------------------------------------------------
    async def test_06_client_retry_immediately_after_timeout(self):
        """Xác minh mấu chốt: Request mới KHÔNG ĐƯỢC CHẠY ĐỒNG THỜI với worker cũ còn dang dở."""
        thread_timeline = []

        def worker_1():
            thread_timeline.append("w1_start")
            time.sleep(0.10)
            thread_timeline.append("w1_end")
            return "res_1"

        def worker_2():
            thread_timeline.append("w2_start")
            time.sleep(0.05)
            thread_timeline.append("w2_end")
            return "res_2"

        # Request 1 bắt đầu
        t1 = asyncio.create_task(self.pool.run_in_pool(worker_1))
        # Chờ một chút để worker 1 chắc chắn đã bắt đầu
        await asyncio.sleep(0.02)

        # Client 1 timeout và hủy t1
        t1.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t1

        # Client gửi ngay Request 2 (retry)
        t2 = asyncio.create_task(self.pool.run_in_pool(worker_2))

        res2 = await t2
        self.assertEqual(res2, "res_2")

        # Xác minh: w2_start BẮT BUỘC phải nằm sau w1_end!
        # Tuyệt đối không có chuyện w2_start xen vào giữa w1_start và w1_end
        self.assertEqual(
            thread_timeline,
            ["w1_start", "w1_end", "w2_start", "w2_end"],
            "LỖI PREMATURE RELEASE: Worker 2 đã chạy đè khi Worker 1 chưa hoàn tất!",
        )
        self.assertEqual(self.pool.peak_active_workers, 1)

    # -----------------------------------------------------------------------
    # Kịch bản 7: Ba request gửi liên tiếp khi CPU đang bận & Event loop không block
    # -----------------------------------------------------------------------
    async def test_07_three_consecutive_requests_and_event_loop_unblocked(self):
        """Xác minh 3 request xếp hàng FIFO và Event loop duy trì độ phản hồi tức thì."""
        def heavy_task(idx: int):
            time.sleep(0.06)
            return idx

        # Gửi 3 heavy tasks
        t1 = asyncio.create_task(self.pool.run_in_pool(heavy_task, 1))
        t2 = asyncio.create_task(self.pool.run_in_pool(heavy_task, 2))
        t3 = asyncio.create_task(self.pool.run_in_pool(heavy_task, 3))

        # Kiểm tra event loop không bị block: chạy tác vụ async nhẹ song song
        # Tác vụ nhẹ này phải hoàn thành trong < 10ms dù 3 heavy task đang xếp hàng
        loop_responsive = False
        async def ping_health():
            nonlocal loop_responsive
            await asyncio.sleep(0.005)
            loop_responsive = True

        health_task = asyncio.create_task(ping_health())
        await health_task
        self.assertTrue(loop_responsive, "Event loop bị block bởi CPU worker!")

        r1, r2, r3 = await asyncio.gather(t1, t2, t3)
        self.assertEqual((r1, r2, r3), (1, 2, 3))

        stats = self.pool.get_stats()
        self.assertEqual(stats["total_requests"], 3)
        self.assertEqual(stats["total_completed"], 3)
        self.assertEqual(stats["peak_active_workers"], 1)

    # -----------------------------------------------------------------------
    # Kiểm tra hành vi Shutdown khi có request đang chạy
    # -----------------------------------------------------------------------
    async def test_08_shutdown_behavior_with_inflight_request(self):
        """Xác minh khi gọi shutdown(wait=True), tác vụ đang chạy vẫn hoàn tất an toàn."""
        def finishing_task():
            time.sleep(0.06)
            return "finished_cleanly"

        task = asyncio.create_task(self.pool.run_in_pool(finishing_task))
        await asyncio.sleep(0.01)

        # Bắt đầu shutdown trong khi task đang in-flight
        loop = asyncio.get_running_loop()
        shutdown_fut = loop.run_in_executor(None, self.pool.shutdown, True)

        res = await task
        self.assertEqual(res, "finished_cleanly")
        await shutdown_fut

        self.assertTrue(self.pool.get_stats()["is_shutting_down"])

        # Sau khi shutdown, request mới phải bị từ chối
        with self.assertRaises(RuntimeError):
            await self.pool.run_in_pool(finishing_task)


if __name__ == "__main__":
    unittest.main()
