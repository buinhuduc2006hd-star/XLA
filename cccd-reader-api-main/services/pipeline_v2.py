"""Service điều phối toàn diện cho API v2 (End-to-End OCR Pipeline).

Giai đoạn 4: FastAPI Integration.
Kết hợp:
1. YOLOv8n Document Detection (cccd / gplx).
2. OpenCV Document Rectifier (856x540 với 4 góc phối cảnh và safety margin).
3. Document Orientation Correction (0°, 90°, 180°, 270°).
4. Adaptive EasyOCR với Calibrated ROIs, Early Exit, và Bounded Field-Level Fallback.
5. Field Validation & Pydantic JSON response structuring.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Literal, Optional

import cv2
import numpy as np

from schemas import (
    DetectionInfo,
    DocumentV2Response,
    FieldValue,
    ProcessingInfo,
    TimingInfo,
)
from services.adaptive_extractor import extract_adaptive_pipeline_e
from services.document_detector import DocumentDetector
from services.document_orientation import DocumentOrientationCorrector
from services.document_rectifier import DocumentRectifier

logger = logging.getLogger(__name__)

# ===========================================================================
# CÁC NGOẠI LỆ NGHIỆP VỤ CHO PIPELINE V2
# ===========================================================================

class PipelineV2Error(Exception):
    """Lớp cha cho các ngoại lệ trong Pipeline V2."""
    def __init__(self, message: str, error_code: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class NoDocumentDetectedError(PipelineV2Error):
    def __init__(self, message: str = "No document (CCCD/GPLX) detected in the provided image", details: dict[str, Any] | None = None):
        super().__init__(message, "NO_DOCUMENT_DETECTED", details)


class DocumentTypeMismatchError(PipelineV2Error):
    def __init__(self, detected: str, requested: str, confidence: float):
        message = f"Document type mismatch: detected '{detected}' (confidence {confidence:.2f}), but requested '{requested}'"
        super().__init__(message, "DOCUMENT_TYPE_MISMATCH", {
            "detected_type": detected,
            "requested_type": requested,
            "confidence": round(confidence, 4),
        })


class InvalidImageError(PipelineV2Error):
    def __init__(self, message: str = "Invalid or corrupt image content", details: dict[str, Any] | None = None):
        super().__init__(message, "INVALID_IMAGE", details)


# ===========================================================================
# SINGLETON MODEL MANAGERS
# ===========================================================================

_DETECTOR_LOCK = threading.Lock()
_DETECTOR_INSTANCE: Optional[DocumentDetector] = None

_RECTIFIER_LOCK = threading.Lock()
_RECTIFIER_INSTANCE: Optional[DocumentRectifier] = None

_ORIENTATION_LOCK = threading.Lock()
_ORIENTATION_INSTANCE: Optional[DocumentOrientationCorrector] = None


def get_document_detector(model_path: str | Path | None = None) -> DocumentDetector:
    """Singleton getter cho DocumentDetector để tái sử dụng model YOLO giữa các request."""
    global _DETECTOR_INSTANCE
    if _DETECTOR_INSTANCE is None:
        with _DETECTOR_LOCK:
            if _DETECTOR_INSTANCE is None:
                # Tìm đường dẫn model mặc định nếu chưa truyền
                if model_path is None:
                    # Ưu tiên model fine-tune trong cccd-reader-api-main/models/
                    candidates = [
                        Path("models/yolov8n_document.pt"),
                        Path("cccd-reader-api-main/models/yolov8n_document.pt"),
                        Path("yolov8n.pt"),
                    ]
                    for c in candidates:
                        if c.exists():
                            model_path = c
                            break
                logger.info(f"Khởi tạo DocumentDetector singleton với model: {model_path}")
                _DETECTOR_INSTANCE = DocumentDetector(model_path=model_path, conf_threshold=0.25)
    return _DETECTOR_INSTANCE


def get_document_rectifier() -> DocumentRectifier:
    """Singleton getter cho DocumentRectifier."""
    global _RECTIFIER_INSTANCE
    if _RECTIFIER_INSTANCE is None:
        with _RECTIFIER_LOCK:
            if _RECTIFIER_INSTANCE is None:
                _RECTIFIER_INSTANCE = DocumentRectifier(
                    target_size=(856, 540),
                    margin=0.08,
                    min_area_ratio=0.20,
                    max_area_ratio=0.98,
                )
    return _RECTIFIER_INSTANCE


def get_orientation_corrector() -> DocumentOrientationCorrector:
    """Singleton getter cho DocumentOrientationCorrector."""
    global _ORIENTATION_INSTANCE
    if _ORIENTATION_INSTANCE is None:
        with _ORIENTATION_LOCK:
            if _ORIENTATION_INSTANCE is None:
                _ORIENTATION_INSTANCE = DocumentOrientationCorrector(
                    header_height_ratio=0.28,
                    min_match_confidence=0.35,
                )
    return _ORIENTATION_INSTANCE


# ===========================================================================
# HÀM ĐIỀU PHỐI CHÍNH END-TO-END
# ===========================================================================

def process_document_v2(
    image: np.ndarray,
    mode: str = "fast",
    document_type: str = "auto",
    detector: DocumentDetector | None = None,
    rectifier: DocumentRectifier | None = None,
    orientation_corrector: DocumentOrientationCorrector | None = None,
) -> DocumentV2Response:
    """Thực thi toàn bộ pipeline End-to-End V2 và trả về DocumentV2Response.

    Args:
        image: Mảng NumPy BGR đã được giải mã hợp lệ.
        mode: 'fast' (ưu tiên latency) hoặc 'accurate' (tối ưu độ chính xác và fallback).
        document_type: 'auto', 'cccd', hoặc 'gplx'.
        detector: Custom detector (dùng cho testing/mocking).
        rectifier: Custom rectifier (dùng cho testing/mocking).
        orientation_corrector: Custom orientation corrector (dùng cho testing/mocking).

    Returns:
        DocumentV2Response có cấu trúc đầy đủ.
    """
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim != 3:
        raise InvalidImageError("Decoded image is empty or not a 3-channel array")

    mode_norm = str(mode).lower().strip()
    if mode_norm not in ("fast", "accurate"):
        mode_norm = "fast"

    doc_type_req = str(document_type).lower().strip()
    if doc_type_req not in ("auto", "cccd", "gplx"):
        raise PipelineV2Error(
            f"Invalid document_type parameter: '{document_type}'. Expected 'auto', 'cccd', or 'gplx'",
            "INVALID_PARAMETER",
        )

    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Bước 1: YOLOv8n Document Detection
    # -----------------------------------------------------------------------
    t_det_start = time.perf_counter()
    active_detector = detector or get_document_detector()
    det_result = active_detector.detect(image)
    t_det_end = time.perf_counter()
    detection_ms = (t_det_end - t_det_start) * 1000.0

    if not det_result.has_detection or not det_result.top_detection:
        raise NoDocumentDetectedError(
            "No CCCD or GPLX detected in the uploaded image",
            details={"num_detections": 0, "image_shape": list(image.shape[:2])},
        )

    top_det = det_result.top_detection
    detected_type = top_det.class_name.lower()
    if detected_type not in ("cccd", "gplx"):
        # Dự phòng trường hợp class không chuẩn
        detected_type = "cccd"

    # Kiểm tra bộ lọc loại giấy tờ nếu người dùng chỉ định
    if doc_type_req != "auto" and detected_type != doc_type_req:
        raise DocumentTypeMismatchError(
            detected=detected_type,
            requested=doc_type_req,
            confidence=top_det.confidence,
        )

    x1, y1, x2, y2 = top_det.bbox_xyxy
    h_img, w_img = image.shape[:2]
    # Crop YOLO gốc an toàn dùng cho fallback
    yolo_crop = image[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)].copy()

    # -----------------------------------------------------------------------
    # Bước 2: Document Rectification (Nắn phối cảnh 4 góc)
    # -----------------------------------------------------------------------
    t_rect_start = time.perf_counter()
    active_rectifier = rectifier or get_document_rectifier()
    rect_result = active_rectifier.rectify(image, top_det.bbox_xyxy, card_type=detected_type)
    t_rect_end = time.perf_counter()
    rectification_ms = (t_rect_end - t_rect_start) * 1000.0

    rectified_img = rect_result.rectified_image

    # -----------------------------------------------------------------------
    # Bước 3: Orientation Correction (Hiệu chỉnh chiều đọc)
    # -----------------------------------------------------------------------
    t_orient_start = time.perf_counter()
    active_orient = orientation_corrector or get_orientation_corrector()
    # Ở chế độ fast: chỉ quét 0° và 180° để tiết kiệm CPU
    check_all = (mode_norm == "accurate")
    orient_result = active_orient.correct_orientation(
        rectified_img,
        card_type=detected_type,
        check_all_angles=check_all,
    )
    t_orient_end = time.perf_counter()
    orientation_ms = (t_orient_end - t_orient_start) * 1000.0

    upright_img = orient_result.image

    # -----------------------------------------------------------------------
    # Bước 4: Adaptive EasyOCR Extraction & Validation
    # -----------------------------------------------------------------------
    t_ocr_start = time.perf_counter()
    ocr_result = extract_adaptive_pipeline_e(
        rectified_image=upright_img,
        yolo_crop=yolo_crop,
        card_type=detected_type,
        mode=mode_norm,
    )
    t_ocr_end = time.perf_counter()
    ocr_ms = (t_ocr_end - t_ocr_start) * 1000.0

    total_ms = (time.perf_counter() - t_start) * 1000.0

    # -----------------------------------------------------------------------
    # Bước 5: Đóng gói Response Schema
    # -----------------------------------------------------------------------
    fields_dict: dict[str, FieldValue] = {}
    for fld, val in ocr_result.fields.items():
        is_valid = ocr_result.valid_flags.get(fld, False)
        src = ocr_result.fallback_sources.get(fld, "rectified" if val else "unextracted")
        fields_dict[fld] = FieldValue(
            value=val,
            valid=is_valid,
            confidence=None,  # EasyOCR heuristic OCR doesn't provide calibrated word-level conf for parsed fields
            source=src or "rectified",
        )

    response = DocumentV2Response(
        success=True,
        document_type=detected_type,
        mode=mode_norm,
        detection=DetectionInfo(
            confidence=round(top_det.confidence, 4),
            bbox=list(top_det.bbox_xyxy),
            num_detections=len(det_result.detections),
            class_name=detected_type,
        ),
        processing=ProcessingInfo(
            rectification_success=rect_result.success,
            rectification_method=rect_result.method,
            orientation_degrees=orient_result.rotation_deg,
            orientation_fallback=orient_result.fallback_used,
            fallback_used=rect_result.fallback_used or orient_result.fallback_used or (ocr_result.total_fallbacks > 0),
        ),
        fields=fields_dict,
        timing_ms=TimingInfo(
            detection_ms=round(detection_ms, 2),
            rectification_ms=round(rectification_ms, 2),
            orientation_ms=round(orientation_ms, 2),
            ocr_ms=round(ocr_ms, 2),
            total_ms=round(total_ms, 2),
        ),
    )

    return response

