"""Module phát hiện và định vị giấy tờ (CCCD & GPLX) sử dụng Ultralytics YOLOv8n Detection.

Hỗ trợ:
- Nhận diện 2 lớp giấy tờ: 0 = 'cccd', 1 = 'gplx'.
- Input: Đường dẫn ảnh (str | Path) hoặc NumPy array BGR (np.ndarray).
- Output: Danh sách detection gồm (class_name, class_id, confidence, bbox_xyxy).
- Inference: Tự động dùng CUDA nếu có, mặc định CPU; hỗ trợ điều chỉnh conf và iou.
- Tiện ích: Vẽ bounding box và lưu ảnh kết quả trực quan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Định nghĩa các lớp mục tiêu chuẩn
CLASS_ID_CCCD = 0
CLASS_ID_GPLX = 1

CLASS_NAMES: dict[int, str] = {
    CLASS_ID_CCCD: "cccd",
    CLASS_ID_GPLX: "gplx",
}

# Bảng màu hiển thị BGR cho từng lớp
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "cccd": (46, 139, 87),     # Sea Green
    "gplx": (218, 112, 37),    # Steel Blue / Golden
    "default": (0, 165, 255),  # Orange
}


@dataclass
class Detection:
    """Biểu diễn một bounding box giấy tờ được phát hiện."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]  # (x1, y1, x2, y2) tọa độ pixel nguyên

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox_xyxy": list(self.bbox_xyxy),
        }


@dataclass
class DetectionResult:
    """Kết quả phát hiện tổng thể trên một ảnh."""

    detections: list[Detection]
    image_shape: tuple[int, int]  # (height, width)
    device: str
    has_detection: bool
    top_detection: Detection | None

    def to_dict(self) -> dict:
        return {
            "has_detection": self.has_detection,
            "count": len(self.detections),
            "image_shape": list(self.image_shape),
            "device": self.device,
            "top_detection": self.top_detection.to_dict() if self.top_detection else None,
            "detections": [d.to_dict() for d in self.detections],
        }


class DocumentDetector:
    """Bộ phát hiện tài liệu CCCD/GPLX sử dụng YOLOv8n."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | None = None,
    ) -> None:
        """Khởi tạo detector.

        Args:
            model_path: Đường dẫn tới trọng số YOLO (.pt). Nếu None, ưu tiên tìm model custom
                        đã train, nếu không có sẽ nạp trọng số nền tảng 'yolov8n.pt'.
            conf_threshold: Ngưỡng confidence tối thiểu (mặc định 0.25).
            iou_threshold: Ngưỡng IoU cho Non-Maximum Suppression (NMS, mặc định 0.45).
            device: Thiết bị chạy ('cpu', 'cuda', hoặc None để tự phát hiện).
        """
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)

        # Tự động chọn thiết bị: ưu tiên CUDA nếu khả dụng, fallback sang CPU
        if device is not None:
            self.device = str(device)
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Xác định đường dẫn model
        self.model_path = self._resolve_model_path(model_path)
        logger.info(f"Loading YOLO model from: {self.model_path} on device: {self.device}")

        # Tải mô hình YOLO
        self.model = YOLO(str(self.model_path))

        # Kiểm tra xem mô hình có phải là fine-tuned cho cccd/gplx không
        self.is_custom_trained = self._check_custom_classes()
        if not self.is_custom_trained:
            logger.warning(
                "Model hiện tại chưa được fine-tune trên tập CCCD/GPLX (chứa classes mặc định). "
                "Cần chạy script scripts/train/train.py để fine-tune trọng số riêng."
            )

    def _resolve_model_path(self, model_path: str | Path | None) -> Path:
        """Tìm đường dẫn model khả dụng."""
        if model_path is not None:
            p = Path(model_path)
            if p.exists():
                return p
            # Nếu người dùng chỉ định đường dẫn cụ thể có thư mục mà không tồn tại
            if "/" in str(model_path) or "\\" in str(model_path):
                raise FileNotFoundError(f"Model file not found: {p}")
            # Nếu là tên file hoặc model Ultralytics chuẩn (vd: 'yolov8n.pt', 'dummy.pt')
            return p

        # Các vị trí ưu tiên tìm model đã train
        candidates = [
            Path("runs/detect/train_document/weights/best.pt"),
            Path("runs/detect/train/weights/best.pt"),
            Path("models/yolov8n_document.pt"),
            Path("cccd-reader-api-main/models/yolov8n_document.pt"),
            Path("yolov8n.pt"),
        ]
        for c in candidates:
            if c.exists():
                return c

        # Mặc định tải/sử dụng yolov8n.pt
        return Path("yolov8n.pt")

    def _check_custom_classes(self) -> bool:
        """Kiểm tra xem model có classes đúng chuẩn 0=cccd, 1=gplx không."""
        try:
            names = self.model.names
            if isinstance(names, dict):
                has_cccd = any("cccd" in str(v).lower() for v in names.values())
                has_gplx = any("gplx" in str(v).lower() for v in names.values())
                return has_cccd or has_gplx
        except Exception:
            pass
        return False

    @staticmethod
    def _validate_image_input(image: Union[str, Path, np.ndarray]) -> tuple[np.ndarray, tuple[int, int]]:
        """Kiểm tra và chuẩn hóa ảnh đầu vào thành NumPy BGR array hợp lệ."""
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Image path does not exist: {img_path}")
            if not img_path.is_file():
                raise ValueError(f"Path is not a regular file: {img_path}")

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise ValueError(f"Failed to read/decode image from: {img_path}")
        elif isinstance(image, np.ndarray):
            bgr = image
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. Expected str, Path, or np.ndarray."
            )

        if bgr.size == 0 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(
                f"Invalid image array format: shape={bgr.shape if hasattr(bgr, 'shape') else None}. "
                "Must be a non-empty 3-channel BGR image."
            )

        h, w = bgr.shape[:2]
        if h < 10 or w < 10:
            raise ValueError(f"Image dimensions too small ({w}x{h}). Minimum required: 10x10.")

        return bgr, (h, w)

    def detect(self, image: Union[str, Path, np.ndarray]) -> DetectionResult:
        """Phát hiện giấy tờ trong ảnh.

        Args:
            image: Đường dẫn ảnh hoặc mảng NumPy BGR.

        Returns:
            DetectionResult chứa danh sách detection và top_detection.
        """
        bgr, (h, w) = self._validate_image_input(image)

        # Thực thi YOLO predict
        results = self.model.predict(
            source=bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()

                for i in range(len(confs)):
                    c_id = int(clss[i])
                    conf = float(confs[i])
                    box = xyxy[i]

                    # Kẹp tọa độ trong giới hạn kích thước ảnh
                    x1 = max(0, min(w - 1, int(round(box[0]))))
                    y1 = max(0, min(h - 1, int(round(box[1]))))
                    x2 = max(0, min(w - 1, int(round(box[2]))))
                    y2 = max(0, min(h - 1, int(round(box[3]))))

                    # Kiểm tra bounding box hợp lệ (chiều dài, rộng > 0)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Xác định tên class
                    if hasattr(self.model, "names") and c_id in self.model.names:
                        c_name = str(self.model.names[c_id])
                    else:
                        c_name = CLASS_NAMES.get(c_id, f"class_{c_id}")

                    detections.append(
                        Detection(
                            class_id=c_id,
                            class_name=c_name,
                            confidence=conf,
                            bbox_xyxy=(x1, y1, x2, y2),
                        )
                    )

        # Sắp xếp theo confidence giảm dần
        detections.sort(key=lambda d: d.confidence, reverse=True)

        top_det = detections[0] if detections else None
        return DetectionResult(
            detections=detections,
            image_shape=(h, w),
            device=self.device,
            has_detection=len(detections) > 0,
            top_detection=top_det,
        )

    @staticmethod
    def annotate_image(
        image: Union[str, Path, np.ndarray],
        detections: list[Detection],
        save_path: str | Path | None = None,
        draw_labels: bool = True,
    ) -> np.ndarray:
        """Vẽ bounding box và nhãn lên ảnh.

        Args:
            image: Ảnh gốc (đường dẫn hoặc NumPy array BGR).
            detections: Danh sách các detection cần vẽ.
            save_path: Đường dẫn lưu ảnh nếu muốn ghi ra file.
            draw_labels: Có vẽ nhãn text và confidence hay không.

        Returns:
            Ảnh NumPy BGR đã được vẽ bounding boxes.
        """
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise ValueError(f"Cannot read image from: {image}")
        else:
            bgr = image.copy()

        h, w = bgr.shape[:2]
        line_thickness = max(2, int(round(min(h, w) * 0.003)))

        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            color = CLASS_COLORS.get(det.class_name.lower(), CLASS_COLORS["default"])

            # Vẽ bounding box
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, line_thickness)

            if draw_labels:
                label_text = f"{det.class_name} {det.confidence:.2f}"
                font_scale = max(0.5, line_thickness * 0.25)
                font_thickness = max(1, line_thickness // 2)

                (tw, th), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )

                # Vẽ nền cho label text
                label_y1 = max(0, y1 - th - baseline - 4)
                label_y2 = y1
                label_x2 = min(w, x1 + tw + 6)
                cv2.rectangle(bgr, (x1, label_y1), (label_x2, label_y2), color, -1)

                # Vẽ text màu trắng tương phản
                cv2.putText(
                    bgr,
                    label_text,
                    (x1 + 3, label_y2 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA,
                )

        if save_path is not None:
            out_p = Path(save_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), bgr)
            logger.info(f"Saved annotated image to: {out_p}")

        return bgr


def detect_document(
    image: Union[str, Path, np.ndarray],
    model_path: str | Path | None = None,
    conf_threshold: float = 0.25,
    device: str | None = None,
) -> DetectionResult:
    """Hàm tiện ích cấp module để phát hiện tài liệu nhanh chóng."""
    detector = DocumentDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        device=device,
    )
    return detector.detect(image)


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="YOLOv8n Document Detector CLI")
    parser.add_argument("image", help="Đường dẫn file ảnh đầu vào")
    parser.add_argument("--model", default=None, help="Đường dẫn trọng số model .pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence (0.0-1.0)")
    parser.add_argument("--save", default=None, help="Đường dẫn lưu ảnh có vẽ bounding box")
    parser.add_argument("--device", default=None, help="Thiết bị ('cpu' hoặc 'cuda')")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    detector = DocumentDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device,
    )

    result = detector.detect(args.image)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    if args.save:
        detector.annotate_image(args.image, result.detections, save_path=args.save)
        print(f"Annotated image saved to: {args.save}")
