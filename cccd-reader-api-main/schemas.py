"""Pydantic schemas cho API v2 của hệ thống CCCD/GPLX Reader.

Định nghĩa schema chuẩn cho response thành công và response lỗi có cấu trúc.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class DetectionInfo(BaseModel):
    """Thông tin về kết quả phát hiện giấy tờ từ YOLOv8n."""

    confidence: float = Field(..., description="Độ tin cậy của bounding box (0.0 -> 1.0)")
    bbox: list[int] = Field(..., description="Tọa độ pixel [x1, y1, x2, y2] trên ảnh gốc")
    num_detections: int = Field(..., description="Tổng số đối tượng giấy tờ phát hiện được trong ảnh")
    class_name: str = Field(..., description="Tên lớp giấy tờ phát hiện được ('cccd' hoặc 'gplx')")


class ProcessingInfo(BaseModel):
    """Thông tin về các bước xử lý hình học (Rectification & Orientation)."""

    rectification_success: bool = Field(..., description="Nắn phối cảnh 4 góc thành công hay không")
    rectification_method: str = Field(..., description="Phương pháp nắn chỉnh ('contour_approx', 'convex_hull', 'fallback_crop'...)")
    orientation_degrees: int = Field(..., description="Góc xoay hiệu chỉnh đọc văn bản (0, 90, 180, 270)")
    orientation_fallback: bool = Field(..., description="Có phải kích hoạt fallback giữ nguyên góc 0° hay không")
    fallback_used: bool = Field(..., description="Bất kỳ bước xử lý hình học nào phải dùng fallback hay không")


class FieldValue(BaseModel):
    """Giá trị và trạng thái xác thực của một trường thông tin trích xuất."""

    value: Optional[str] = Field(None, description="Chuỗi văn bản trích xuất được (hoặc null nếu không tìm thấy)")
    valid: bool = Field(False, description="Kết quả có thỏa mãn validator nghiệp vụ định dạng hay không")
    confidence: Optional[float] = Field(None, description="Độ tin cậy OCR của trường (nếu có)")
    source: str = Field("rectified", description="Nguồn trích xuất ('rectified', 'yolo_crop_fallback', 'unextracted')")


class TimingInfo(BaseModel):
    """Thời gian xử lý từng công đoạn tính bằng mili-giây (ms)."""

    detection_ms: float = Field(..., description="Thời gian phát hiện YOLOv8n (ms)")
    rectification_ms: float = Field(..., description="Thời gian nắn phối cảnh OpenCV (ms)")
    orientation_ms: float = Field(..., description="Thời gian xác định chiều văn bản (ms)")
    ocr_ms: float = Field(..., description="Thời gian trích xuất EasyOCR (ms)")
    total_ms: float = Field(..., description="Tổng thời gian end-to-end (ms)")


class DocumentV2Response(BaseModel):
    """Schema response chuẩn trả về từ POST /api/v2/upload."""

    success: bool = Field(True, description="Trạng thái xử lý thành công")
    document_type: str = Field(..., description="Loại giấy tờ ('cccd' hoặc 'gplx')")
    mode: str = Field(..., description="Chế độ xử lý ('fast' hoặc 'accurate')")
    detection: DetectionInfo = Field(..., description="Chi tiết phát hiện vị trí giấy tờ")
    processing: ProcessingInfo = Field(..., description="Chi tiết nắn phối cảnh và định hướng")
    fields: dict[str, FieldValue] = Field(..., description="Các trường thông tin đã trích xuất")
    timing_ms: TimingInfo = Field(..., description="Thời gian xử lý từng công đoạn")


class ErrorResponse(BaseModel):
    """Schema response chuẩn khi có lỗi nghiệp vụ hoặc kỹ thuật."""

    success: bool = Field(False, description="Trạng thái thất bại")
    error_code: str = Field(..., description="Mã lỗi có cấu trúc (ví dụ: NO_DOCUMENT_DETECTED, TYPE_MISMATCH)")
    message: str = Field(..., description="Thông báo lỗi mô tả ngắn gọn")
    details: Optional[dict[str, Any]] = Field(None, description="Chi tiết bổ sung về lỗi (nếu có)")

