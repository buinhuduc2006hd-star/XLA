"""Service điều phối pipeline Pose-only (YOLOv8-Pose End-to-End).

Kiến trúc Pose-only:
Input Image
    |
    v
YOLOv8n-Pose (Single Inference: Document Type + Bounding Box + 4 Physical Keypoints)
    |
    v
Keypoint Geometric Validation (Convexity, Area, Bounds, Clockwise Order, Confidences)
    |
    v
Direct Perspective Rectification (cv2.getPerspectiveTransform + warpPerspective -> 856x540)
    |
    v
Document Orientation Correction (0°, 90°, 180°, 270°)
    |
    v
Calibrated ROI Extraction & Adaptive EasyOCR
    |
    v
Field Cleaning & Validation
    |
    v
Structured JSON (DocumentV2Response)

Ưu điểm cốt lõi so với Legacy:
1. Chỉ chạy 1 lần suy luận YOLO duy nhất trên toàn bộ ảnh.
2. Không cần chạy lại mô hình YOLO Detection riêng biệt.
3. Không cần các bước heuristic OpenCV contour/edge detection tốn kém và nhạy cảm với chói sáng/nền phức tạp.
4. Trả lỗi rõ ràng và không thực thi OCR lãng phí nếu góc thẻ không hợp lệ.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

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
from services.document_orientation import (
    DocumentOrientationCorrector,
    get_shared_easyocr_reader,
)
from services.document_pose_detector import (
    DocumentPoseDetector,
    KeypointValidationError,
    PoseDetectionResult,
    PoseDetectorError,
    PoseModelIncompatibleError,
    PoseModelNotFoundError,
    get_document_pose_detector,
)
from services.document_rectifier import DocumentRectifier

logger = logging.getLogger(__name__)


# ==============================================================================
# EXCEPTIONS CHO PIPELINE POSE
# ==============================================================================

class PipelinePoseError(Exception):
    """Lớp cha cho các ngoại lệ trong Pipeline Pose."""
    def __init__(self, message: str, error_code: str = "PIPELINE_ERROR", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class PipelineV2Error(PipelinePoseError):
    """Alias tương thích ngược cho PipelineV2Error."""
    pass


class NoDocumentDetectedError(PipelinePoseError):
    def __init__(self, message: str = "No document (CCCD/GPLX) detected in the provided image", details: dict[str, Any] | None = None):
        super().__init__(message, "NO_DOCUMENT_DETECTED", details)


class DocumentTypeMismatchError(PipelinePoseError):
    def __init__(self, detected: str, requested: str, confidence: float):
        message = f"Document type mismatch: detected '{detected}' (confidence {confidence:.2f}), but requested '{requested}'"
        super().__init__(message, "DOCUMENT_TYPE_MISMATCH", {
            "detected_type": detected,
            "requested_type": requested,
            "confidence": round(confidence, 4),
        })


class InvalidImageError(PipelinePoseError):
    def __init__(self, message: str = "Invalid or corrupt image content", details: dict[str, Any] | None = None):
        super().__init__(message, "INVALID_IMAGE", details)


# ==============================================================================
# SINGLETON MANAGERS CHO RECTIFIER & ORIENTATION
# ==============================================================================

_RECTIFIER_LOCK = threading.Lock()
_RECTIFIER_INSTANCE: Optional[DocumentRectifier] = None

_ORIENTATION_LOCK = threading.Lock()
_ORIENTATION_INSTANCE: Optional[DocumentOrientationCorrector] = None


def get_document_rectifier() -> DocumentRectifier:
    """Singleton getter cho DocumentRectifier."""
    global _RECTIFIER_INSTANCE
    if _RECTIFIER_INSTANCE is None:
        with _RECTIFIER_LOCK:
            if _RECTIFIER_INSTANCE is None:
                _RECTIFIER_INSTANCE = DocumentRectifier(
                    target_size=(856, 540),
                    margin=0.08,
                    min_area_ratio=0.05,
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


def process_document_pose(
    image: np.ndarray,
    mode: str = "fast",
    document_type: str = "auto",
    pose_detector: DocumentPoseDetector | None = None,
    rectifier: DocumentRectifier | None = None,
    orientation_corrector: DocumentOrientationCorrector | None = None,
) -> DocumentV2Response:
    """Thực thi toàn bộ pipeline Pose-only End-to-End.

    Args:
        image: Mảng NumPy BGR 3 kênh hợp lệ.
        mode: 'fast' (nhanh) hoặc 'accurate' (kiểm tra toàn diện).
        document_type: 'auto', 'cccd', hoặc 'gplx'.
        pose_detector: Custom pose detector (dùng cho testing/mocking).
        rectifier: Custom rectifier (dùng cho testing/mocking).
        orientation_corrector: Custom orientation corrector (dùng cho testing/mocking).

    Returns:
        DocumentV2Response với pipeline_mode='pose' và thông tin 4 keypoints.
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
    # Bước 1: YOLOv8-Pose Document & 4-Corner Detection
    # -----------------------------------------------------------------------
    t_pose_start = time.perf_counter()
    active_detector = pose_detector or get_document_pose_detector()
    pose_res: PoseDetectionResult = active_detector.detect(
        image=image,
        expected_type=doc_type_req,
    )
    t_pose_end = time.perf_counter()
    pose_ms = (t_pose_end - t_pose_start) * 1000.0

    if not pose_res.success:
        reason = pose_res.failure_reason or "Pose detection failed"
        logger.warning(f"Pose detection rejected: {reason}")
        if "No document detected" in reason or "0 bounding boxes" in reason:
            raise NoDocumentDetectedError(reason, details=pose_res.to_dict())
        if "No document matching requested type" in reason:
            raise DocumentTypeMismatchError(
                detected=pose_res.document_type or "none",
                requested=doc_type_req,
                confidence=pose_res.confidence or 0.0,
            )
        if "Keypoint validation failed" in reason:
            raise KeypointValidationError(reason, details=pose_res.to_dict())
        raise PipelineV2Error(reason, "POSE_DETECTION_FAILED", details=pose_res.to_dict())

    detected_type = pose_res.document_type or "cccd"
    if doc_type_req != "auto" and detected_type != doc_type_req:
        raise DocumentTypeMismatchError(
            detected=detected_type,
            requested=doc_type_req,
            confidence=pose_res.confidence or 0.0,
        )

    # -----------------------------------------------------------------------
    # Bước 2: Direct Perspective Rectification từ 4 Keypoints
    # -----------------------------------------------------------------------
    t_rect_start = time.perf_counter()
    active_rectifier = rectifier or get_document_rectifier()
    rect_result = active_rectifier.rectify_from_corners(
        image=image,
        corners=pose_res.corners,
        card_type=detected_type,
        target_size=(856, 540),
    )
    t_rect_end = time.perf_counter()
    rectification_ms = (t_rect_end - t_rect_start) * 1000.0

    if not rect_result.success:
        raise PipelineV2Error(
            f"Failed to rectify document from predicted corners: {rect_result.error}",
            "RECTIFICATION_FAILED",
            details=rect_result.to_dict(),
        )

    rectified_img = rect_result.rectified_image

    # -----------------------------------------------------------------------
    # Bước 3: Orientation Correction (Hiệu chỉnh chiều đọc văn bản)
    # -----------------------------------------------------------------------
    t_orient_start = time.perf_counter()
    active_orient = orientation_corrector or get_orientation_corrector()
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
    # Bước 4: Calibrated ROI Extraction & Adaptive EasyOCR
    # -----------------------------------------------------------------------
    t_ocr_start = time.perf_counter()
    # Ở chế độ Pose: đã nắn phẳng góc chuẩn, không cần yolo_crop méo
    ocr_result = extract_adaptive_pipeline_e(
        rectified_image=upright_img,
        yolo_crop=None,
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
    raw_ocr_dict: dict[str, Optional[str]] = dict(ocr_result.raw_fields)
    validation_dict: dict[str, bool] = dict(ocr_result.valid_flags)

    for fld, val in ocr_result.fields.items():
        is_valid = ocr_result.valid_flags.get(fld, False)
        is_clean = ocr_result.clean_flags.get(fld, True)
        raw_val = ocr_result.raw_fields.get(fld)
        src = ocr_result.fallback_sources.get(fld, "pose_rectified" if val else "unextracted")
        fields_dict[fld] = FieldValue(
            value=val,
            raw_value=raw_val,
            valid=is_valid,
            clean=is_clean,
            confidence=None,
            source=src or "pose_rectified",
        )
        validation_dict[fld] = is_valid
        if fld not in raw_ocr_dict:
            raw_ocr_dict[fld] = raw_val

    corners_list = pose_res.corners.astype(int).tolist() if pose_res.corners is not None else None

    response = DocumentV2Response(
        success=True,
        document_type=detected_type,
        mode=mode_norm,
        pipeline_mode="pose",
        detection=DetectionInfo(
            confidence=round(pose_res.confidence or 0.0, 4),
            bbox=list(pose_res.bbox) if pose_res.bbox else [0, 0, 0, 0],
            num_detections=1,
            class_name=detected_type,
        ),
        processing=ProcessingInfo(
            rectification_success=rect_result.success,
            rectification_method=rect_result.method,
            orientation_degrees=orient_result.rotation_deg,
            orientation_fallback=orient_result.fallback_used,
            fallback_used=rect_result.fallback_used or orient_result.fallback_used or (ocr_result.total_fallbacks > 0),
        ),
        keypoints=corners_list,
        keypoint_confidences=pose_res.keypoint_confidences,
        corner_validation_status="valid",
        fields=fields_dict,
        raw_ocr=raw_ocr_dict,
        validation=validation_dict,
        timing_ms=TimingInfo(
            detection_ms=round(pose_ms, 2),
            rectification_ms=round(rectification_ms, 2),
            orientation_ms=round(orientation_ms, 2),
            ocr_ms=round(ocr_ms, 2),
            total_ms=round(total_ms, 2),
        ),
    )

    return response
