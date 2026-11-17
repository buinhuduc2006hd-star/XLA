"""Module điều khiển Concurrency & Thread Pool cho quá trình suy luận AI (Giai đoạn 6).

Mục tiêu cốt lõi:
1. Kiểm soát số lượng worker thực thi suy luận CPU đồng thời (mặc định = 1) để bảo vệ RAM/CPU.
2. Ngăn ngừa triệt để lỗi "giải phóng sớm" (Premature Semaphore Release):
   Khi client ngắt kết nối hoặc timeout, tác vụ asyncio bị cancel nhưng luồng worker bên dưới
   vẫn đang chạy. Semaphore TUYỆT ĐỐI KHÔNG ĐƯỢC GIẢI PHÓNG cho đến khi luồng worker thực sự kết thúc.
3. Non-blocking Event Loop: Mọi thao tác chờ đợi slot được thực hiện bất đồng bộ trên Event Loop,
   giữ cho các endpoint nhẹ (/health, /docs) luôn phản hồi tức thì (< 5ms).
4. FIFO Queue: Các request đến sau sẽ chờ theo hàng đợi và được phục vụ tuần tự khi worker rảnh.
5. Graceful Shutdown: Đợi tác vụ đang chạy hoàn tất khi ứng dụng tắt, không gây hỏng trạng thái.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class InferencePool:
    """Quản lý luồng thực thi tác vụ suy luận AI với cơ chế đồng bộ an toàn tuyệt đối."""

    def __init__(self, max_workers: int = 1):
        if max_workers < 1:
            raise ValueError("max_workers phải lớn hơn hoặc bằng 1")

        self.max_workers = max_workers
        self.active_workers = 0
        self.peak_active_workers = 0
        self.total_completed = 0
        self.total_cancelled = 0
        self.total_requests = 0

        self._lock = threading.Lock()
        self._semaphore: asyncio.Semaphore | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="ocr_inference",
        )
        self._is_shutting_down = False

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Khởi tạo Semaphore gắn với Event Loop hiện tại khi được gọi lần đầu."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_workers)
        return self._semaphore

    async def run_in_pool(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Thực thi một callable đồng bộ trong thread pool với cơ chế bảo vệ concurrency.

        Nếu coroutine của request bị hủy (client disconnect/timeout), slot của worker
        chỉ được giải phóng sau khi luồng worker thực sự kết thúc tính toán.
        """
        if self._is_shutting_down:
            raise RuntimeError("InferencePool đang tắt, không nhận thêm request mới")

        sem = self._get_semaphore()

        # 1. Chờ lấy slot trong asyncio (không block event loop)
        await sem.acquire()

        with self._lock:
            self.total_requests += 1

        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()

        def _worker_wrapper():
            """Wrapper chạy trong worker thread."""
            with self._lock:
                self.active_workers += 1
                if self.active_workers > self.peak_active_workers:
                    self.peak_active_workers = self.active_workers

            try:
                result = func(*args, **kwargs)
                if not future.done():
                    loop.call_soon_threadsafe(future.set_result, result)
            except BaseException as exc:
                if not future.done():
                    loop.call_soon_threadsafe(future.set_exception, exc)
            finally:
                with self._lock:
                    self.active_workers -= 1
                    self.total_completed += 1

                # QUAN TRỌNG NHẤT: Giải phóng semaphore CHỈ KHI worker thread thực sự kết thúc
                loop.call_soon_threadsafe(sem.release)

        # 2. Đưa tác vụ vào ThreadPoolExecutor
        self._executor.submit(_worker_wrapper)

        # 3. Chờ kết quả từ worker thread
        try:
            return await future
        except asyncio.CancelledError:
            with self._lock:
                self.total_cancelled += 1
            # coroutine bị cancel (do client timeout hoặc disconnect),
            # re-raise để FastAPI phản hồi client, nhưng slot semaphore
            # sẽ do _worker_wrapper.finally giải phóng khi thread xong.
            raise

    def get_stats(self) -> dict[str, Any]:
        """Lấy số liệu thống kê vận hành thời gian thực của pool."""
        with self._lock:
            return {
                "max_workers": self.max_workers,
                "active_workers": self.active_workers,
                "peak_active_workers": self.peak_active_workers,
                "total_requests": self.total_requests,
                "total_completed": self.total_completed,
                "total_cancelled": self.total_cancelled,
                "is_shutting_down": self._is_shutting_down,
            }

    def shutdown(self, wait: bool = True) -> None:
        """Đóng pool và dọn dẹp tài nguyên gracefully."""
        self._is_shutting_down = True
        self._executor.shutdown(wait=wait, cancel_futures=True)


# Instance toàn cục duy nhất của InferencePool
_GLOBAL_POOL: InferencePool | None = None
_POOL_INIT_LOCK = threading.Lock()


def get_inference_pool() -> InferencePool:
    """Trả về singleton InferencePool được cấu hình qua biến môi trường MAX_CONCURRENT_INFERENCE."""
    global _GLOBAL_POOL
    if _GLOBAL_POOL is None:
        with _POOL_INIT_LOCK:
            if _GLOBAL_POOL is None:
                max_w = int(os.getenv("MAX_CONCURRENT_INFERENCE", "1"))
                _GLOBAL_POOL = InferencePool(max_workers=max_w)
    return _GLOBAL_POOL


def reset_inference_pool_for_test(max_workers: int = 1) -> InferencePool:
    """Hàm hỗ trợ testing để reset pool với cấu hình mong muốn."""
    global _GLOBAL_POOL
    with _POOL_INIT_LOCK:
        if _GLOBAL_POOL is not None:
            _GLOBAL_POOL.shutdown(wait=False)
        _GLOBAL_POOL = InferencePool(max_workers=max_workers)
    return _GLOBAL_POOL
