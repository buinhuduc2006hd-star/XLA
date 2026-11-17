"""Module phát hiện và định vị 4 góc thẻ giấy tờ bằng YOLOv8-Pose (Pose-only Pipeline).

Nhiệm vụ:
1. Phát hiện và phân loại giấy tờ (cccd / gplx).
2. Dự đoán trực tiếp tọa độ 4 góc vật lý thực tế của thẻ (Top-Left, Top-Right, Bottom-Right, Bottom-Left).
3. Kiểm tra tính hợp lệ hình học (Keypoint Validation) trước khi nắn phẳng.
4. Độc lập hoàn toàn với YOLO Object Detection cũ (không nạp Detection model khi chạy Pose).
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Class ID mapping chuẩn cho mô hình YOLOv8-Pose CCCD & GPLX
CLASS_ID_TO_NAME: dict[int, str] = {
    0: "cccd",
    1: "gplx",
}
NAME_TO_CLASS_ID: dict[str, int] = {v: k for k, v in CLASS_ID_TO_NAME.items()}

# Tên và thứ tự 4 keypoint chuẩn theo chiều kim đồng hồ
KEYPOINT_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]


# ==============================================================================
# CÁC NGOẠI LỆ CHO POSE DETECTOR
# ==============================================================================

class PoseDetectorError(Exception):
    """Lớp cha cho các ngoại lệ liên quan đến Pose Detector."""
    def __init__(self, message: str, error_code: str = "POSE_ERROR", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class PoseModelNotFoundError(PoseDetectorError):
    """Ngoại lệ khi không tìm thấy file checkpoint trọng số YOLOv8-Pose."""
    def __init__(self, model_path: str | Path):
        msg = f"YOLOv8-Pose checkpoint not found at '{model_path}'. Please train or provide a valid model."
        super().__init__(msg, "POSE_MODEL_NOT_FOUND", {"model_path": str(model_path)})


class PoseModelIncompatibleError(PoseDetectorError):
    """Ngoại lệ khi checkpoint không tương thích hoặc không phải mô hình Pose 4 keypoints."""
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, "POSE_MODEL_INCOMPATIBLE", details)


class KeypointValidationError(PoseDetectorError):
    """Ngoại lệ khi 4 góc thẻ không vượt qua các tiêu chuẩn hình học vật lý."""
    def __init__(self, reason: str, details: dict[str, Any] | None = None):
        super().__init__(f"Keypoint validation failed: {reason}", "KEYPOINT_VALIDATION_FAILED", details)


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class PoseDetectionResult:
    """Kết quả phát hiện và dự đoán 4 góc thẻ từ YOLOv8-Pose."""

    success: bool
    document_type: str | None
    bbox: tuple[int, int, int, int] | None  # (x1, y1, x2, y2)
    confidence: float | None
    corners: np.ndarray | None  # shape (4, 2) float32 trong hệ pixel ảnh gốc [TL, TR, BR, BL]
    keypoint_confidences: list[float] | None  # 4 điểm số tin cậy của 4 keypoint
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "document_type": self.document_type,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "corners": self.corners.tolist() if self.corners is not None else None,
            "keypoint_confidences": [round(c, 4) for c in self.keypoint_confidences] if self.keypoint_confidences else None,
            "failure_reason": self.failure_reason,
        }


# ==============================================================================
# KEYPOINT VALIDATOR (NHIỆM VỤ 4)
# ==============================================================================

def validate_card_keypoints(
    corners: np.ndarray,
    image_shape: tuple[int, int],
    keypoint_confidences: list[float] | None = None,
    min_kpt_conf: float = 0.25,
    min_avg_kpt_conf: float = 0.35,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.98,
    min_edge_length_px: float = 25.0,
    margin_tolerance_ratio: float = 0.08,
) -> tuple[bool, str | None]:
    """Kiểm tra tính hợp lệ hình học và vật lý của 4 Keypoint dự đoán bởi YOLOv8-Pose.

    Các ràng buộc kiểm tra:
    1. Đúng định dạng mảng (4, 2) tọa độ số thực.
    2. Tọa độ hữu hạn (không chứa NaN, Inf).
    3. Nằm trong phạm vi khung ảnh (cho phép dung sai ngoại suy nhỏ margin_tolerance_ratio).
    4. Không có điểm trùng lặp, khoảng cách giữa các góc kề nhau >= min_edge_length_px.
    5. Đạt ngưỡng độ tin cậy keypoint (không có góc nào quá mờ/không chắc chắn).
    6. Tứ giác phải là đa giác lồi (Convex polygon), không tự cắt (no self-intersection).
    7. Thứ tự 4 góc theo chiều kim đồng hồ: TL -> TR -> BR -> BL.
    8. Diện tích tứ giác đủ lớn và không vượt quá toàn bộ khung ảnh.

    LƯU Ý: Không bắt buộc mọi góc phải vuông 90° và không bắt buộc tỷ lệ cạnh bằng 1.5858
    trên ảnh gốc vì ảnh thực tế bị méo phối cảnh và nghiêng góc lớn.
    """
    if corners is None:
        return False, "Corners array is None"

    if not isinstance(corners, np.ndarray):
        return False, f"Corners must be numpy array, got {type(corners)}"

    if corners.shape != (4, 2):
        return False, f"Corners shape must be (4, 2), got {corners.shape}"

    # 1. Tọa độ hữu hạn
    if not np.all(np.isfinite(corners)):
        return False, "Corners contain NaN or Inf values"

    h_img, w_img = image_shape[:2]
    if h_img <= 0 or w_img <= 0:
        return False, "Invalid image dimensions"

    # 2. Dung sai giới hạn biên ảnh (cho phép ngoại suy góc bo tròn ở sát mép)
    tol_x = w_img * margin_tolerance_ratio
    tol_y = h_img * margin_tolerance_ratio
    min_x, max_x = -tol_x, w_img + tol_x
    min_y, max_y = -tol_y, h_img + tol_y

    for i, (x, y) in enumerate(corners):
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return False, f"Keypoint {KEYPOINT_NAMES[i]} ({x:.1f}, {y:.1f}) is out of allowed image bounds [-{tol_x:.0f}, {w_img + tol_x:.0f}]"

    # 3. Kiểm tra độ tin cậy của keypoints nếu có
    if keypoint_confidences is not None and len(keypoint_confidences) == 4:
        for i, conf in enumerate(keypoint_confidences):
            if conf < min_kpt_conf:
                return False, f"Keypoint {KEYPOINT_NAMES[i]} confidence ({conf:.2f}) is below minimum threshold ({min_kpt_conf})"
        avg_conf = sum(keypoint_confidences) / 4.0
        if avg_conf < min_avg_kpt_conf:
            return False, f"Average keypoint confidence ({avg_conf:.2f}) is below minimum threshold ({min_avg_kpt_conf})"

    # 4. Kiểm tra độ dài các cạnh và điểm trùng lặp
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        edge_len = float(np.linalg.norm(p1 - p2))
        if edge_len < min_edge_length_px:
            return False, f"Edge {KEYPOINT_NAMES[i]} -> {KEYPOINT_NAMES[(i + 1) % 4]} is too short ({edge_len:.1f}px < {min_edge_length_px}px)"

    # Kiểm tra đường chéo không bị chập (trùng điểm đối diện)
    d1 = float(np.linalg.norm(corners[0] - corners[2]))
    d2 = float(np.linalg.norm(corners[1] - corners[3]))
    if d1 < min_edge_length_px * 1.4 or d2 < min_edge_length_px * 1.4:
        return False, "Opposite corners are collapsed (degenerate quad)"

    # 5. Tứ giác lồi (Convex polygon check)
    poly = corners.astype(np.int32)
    if not cv2.isContourConvex(poly):
        return False, "Quadrilateral is not convex (possible bow-tie or self-intersecting polygon)"

    # 6. Kiểm tra diện tích
    area = cv2.contourArea(poly)
    img_area = float(h_img * w_img)
    area_ratio = area / img_area

    if area_ratio < min_area_ratio:
        return False, f"Quad area ratio ({area_ratio:.3f}) is too small (< {min_area_ratio})"
    if area_ratio > max_area_ratio:
        return False, f"Quad area ratio ({area_ratio:.3f}) is suspiciously large (> {max_area_ratio})"

    # 7. Kiểm tra thứ tự vòng chiều kim đồng hồ (Clockwise ordering via cross-product)
    # Vector TL -> TR và TL -> BL
    v_top = corners[1] - corners[0]   # (dx, dy) từ TL đến TR
    v_left = corners[3] - corners[0]  # (dx, dy) từ TL đến BL
    cross_z = v_top[0] * v_left[1] - v_top[1] * v_left[0]
    if cross_z <= 0:
        return False, "Corners ordering is counter-clockwise or inverted (mirrored)"

    return True, None


# ==============================================================================
# POSE DETECTOR CLASS (NHIỆM VỤ 3)
# ==============================================================================

class DocumentPoseDetector:
    """Mô hình phát hiện giấy tờ và 4 góc vật lý sử dụng YOLOv8-Pose."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.25,
        kpt_conf_threshold: float = 0.25,
        device: str = "cpu",
    ):
        self.conf_threshold = conf_threshold
        self.kpt_conf_threshold = kpt_conf_threshold
        self.device = device
        self.model_path = self._resolve_model_path(model_path)
        self.model = None
        self._load_lock = threading.Lock()
        self._load_model()

    def _resolve_model_path(self, path: str | Path | None) -> Path:
        """Tìm đường dẫn checkpoint mô hình Pose."""
        if path:
            p = Path(path)
            if p.exists():
                return p.resolve()
            raise PoseModelNotFoundError(p)

        # Kiểm tra biến môi trường
        env_path = os.getenv("POSE_MODEL_PATH")
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p.resolve()

        # Kiểm tra các vị trí mặc định trong project
        root = Path(__file__).resolve().parent.parent
        candidates = [
            root / "models" / "document_pose" / "best.pt",
            root / "models" / "yolov8n_pose_document.pt",
            root / "models" / "best_pose.pt",
            Path("models/document_pose/best.pt"),
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()

        # Nếu không tìm thấy, chỉ định vị trí kỳ vọng chuẩn
        expected = root / "models" / "document_pose" / "best.pt"
        raise PoseModelNotFoundError(expected)

    def _load_model(self) -> None:
        """Nạp checkpoint Ultralytics YOLOv8-Pose vào bộ nhớ."""
        with self._load_lock:
            if self.model is not None:
                return
            logger.info(f"Đang nạp mô hình YOLOv8-Pose từ: {self.model_path} trên thiết bị: {self.device}")
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(self.model_path))
                # Xác minh model có đầu ra keypoints
                if hasattr(self.model, "task") and self.model.task != "pose":
                    logger.warning(f"Cảnh báo: Model task là '{self.model.task}', kỳ vọng 'pose'")
            except Exception as e:
                logger.error(f"Không thể nạp mô hình YOLOv8-Pose: {e}")
                raise PoseModelIncompatibleError(f"Failed to initialize YOLOv8-Pose model from '{self.model_path}': {e}")

    def detect(
        self,
        image: np.ndarray,
        expected_type: str = "auto",
        conf: float | None = None,
    ) -> PoseDetectionResult:
        """Thực thi dự đoán Bounding Box và 4 Keypoints trên ảnh đầu vào.

        Args:
            image: Ảnh BGR NumPy array.
            expected_type: 'auto', 'cccd', hoặc 'gplx'.
            conf: Ngưỡng confidence tối thiểu (nếu None, dùng mặc định).

        Returns:
            PoseDetectionResult chứa kết quả phát hiện hoặc lỗi chi tiết.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return PoseDetectionResult(
                success=False,
                document_type=None,
                bbox=None,
                confidence=None,
                corners=None,
                keypoint_confidences=None,
                failure_reason="Invalid or empty input image matrix",
            )

        threshold = conf if conf is not None else self.conf_threshold

        try:
            results = self.model.predict(
                source=image,
                conf=threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Lỗi khi suy luận YOLOv8-Pose: {e}")
            return PoseDetectionResult(
                success=False,
                document_type=None,
                bbox=None,
                confidence=None,
                corners=None,
                keypoint_confidences=None,
                failure_reason=f"Inference error: {e}",
            )

        if not results or len(results) == 0:
            return PoseDetectionResult(
                success=False,
                document_type=None,
                bbox=None,
                confidence=None,
                corners=None,
                keypoint_confidences=None,
                failure_reason="No detections produced by YOLOv8-Pose",
            )

        res = results[0]
        boxes = res.boxes
        keypoints = res.keypoints

        if boxes is None or len(boxes) == 0:
            return PoseDetectionResult(
                success=False,
                document_type=None,
                bbox=None,
                confidence=None,
                corners=None,
                keypoint_confidences=None,
                failure_reason="No document detected (0 bounding boxes)",
            )

        if keypoints is None or len(keypoints) == 0:
            return PoseDetectionResult(
                success=False,
                document_type=None,
                bbox=None,
                confidence=None,
                corners=None,
                keypoint_confidences=None,
                failure_reason="Model did not output keypoints (model may not be pose-type)",
            )

        # Chuyển đổi tensor sang NumPy
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        boxes_conf = boxes.conf.cpu().numpy()
        boxes_cls = boxes.cls.cpu().numpy().astype(int)

        kpts_xy = keypoints.xy.cpu().numpy()  # [N, 4, 2] tọa độ pixel gốc
        kpts_conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None  # [N, 4]

        # Multi-document selection: Chọn đối tượng tốt nhất
        # Tiêu chí:
        # 1. Khớp class yêu cầu nếu expected_type != 'auto'
        # 2. Score = box_conf * sqrt(box_area)
        best_idx = -1
        best_score = -1.0
        req_cls_id = NAME_TO_CLASS_ID.get(expected_type.lower()) if expected_type != "auto" else None

        for idx in range(len(boxes_xyxy)):
            cls_id = int(boxes_cls[idx])
            # Bỏ qua nếu không đúng loại yêu cầu
            if req_cls_id is not None and cls_id != req_cls_id:
                continue

            bx1, by1, bx2, by2 = boxes_xyxy[idx]
            b_area = max(0.0, float((bx2 - bx1) * (by2 - by1)))
            b_conf = float(boxes_conf[idx])
            score = b_conf * math.sqrt(b_area)

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            return PoseDetectionResult(
                success=False,
                document_type=None,
                bbox=None,
                confidence=None,
                corners=None,
                keypoint_confidences=None,
                failure_reason=f"No document matching requested type '{expected_type}'",
            )

        # Trích xuất đối tượng tốt nhất
        best_cls_id = int(boxes_cls[best_idx])
        best_doc_type = CLASS_ID_TO_NAME.get(best_cls_id, f"unknown_{best_cls_id}")
        b_x1, b_y1, b_x2, b_y2 = [int(round(v)) for v in boxes_xyxy[best_idx]]
        best_box_conf = float(boxes_conf[best_idx])

        # 4 Keypoint dự đoán
        best_corners = kpts_xy[best_idx]  # shape (4, 2)
        if best_corners.shape != (4, 2):
            return PoseDetectionResult(
                success=False,
                document_type=best_doc_type,
                bbox=(b_x1, b_y1, b_x2, b_y2),
                confidence=best_box_conf,
                corners=None,
                keypoint_confidences=None,
                failure_reason=f"Malformed keypoint array shape: {best_corners.shape}",
            )

        best_kpt_conf = [float(c) for c in kpts_conf[best_idx]] if kpts_conf is not None else [1.0, 1.0, 1.0, 1.0]

        # Thực thi Keypoint Validation (Nhiệm vụ 4)
        is_valid, val_reason = validate_card_keypoints(
            corners=best_corners,
            image_shape=image.shape[:2],
            keypoint_confidences=best_kpt_conf,
            min_kpt_conf=self.kpt_conf_threshold,
        )

        if not is_valid:
            logger.warning(f"Keypoint validation failed for {best_doc_type}: {val_reason}")
            return PoseDetectionResult(
                success=False,
                document_type=best_doc_type,
                bbox=(b_x1, b_y1, b_x2, b_y2),
                confidence=best_box_conf,
                corners=best_corners,
                keypoint_confidences=best_kpt_conf,
                failure_reason=f"Keypoint validation failed: {val_reason}",
            )

        return PoseDetectionResult(
            success=True,
            document_type=best_doc_type,
            bbox=(b_x1, b_y1, b_x2, b_y2),
            confidence=best_box_conf,
            corners=best_corners.astype(np.float32),
            keypoint_confidences=best_kpt_conf,
            failure_reason=None,
        )


# ==============================================================================
# SINGLETON MODEL ACCESSOR
# ==============================================================================

_POSE_DETECTOR_LOCK = threading.Lock()
_POSE_DETECTOR_INSTANCE: Optional[DocumentPoseDetector] = None


def get_document_pose_detector(model_path: str | Path | None = None) -> DocumentPoseDetector:
    """Singleton getter cho DocumentPoseDetector.

    Đảm bảo mô hình YOLOv8-Pose chỉ khởi tạo duy nhất một lần trên tiến trình.
    """
    global _POSE_DETECTOR_INSTANCE
    if _POSE_DETECTOR_INSTANCE is None:
        with _POSE_DETECTOR_LOCK:
            if _POSE_DETECTOR_INSTANCE is None:
                _POSE_DETECTOR_INSTANCE = DocumentPoseDetector(model_path=model_path)
    return _POSE_DETECTOR_INSTANCE


def reset_document_pose_detector() -> None:
    """Reset singleton (chủ yếu dùng cho unit test hoặc reload checkpoint)."""
    global _POSE_DETECTOR_INSTANCE
    with _POSE_DETECTOR_LOCK:
        _POSE_DETECTOR_INSTANCE = None
