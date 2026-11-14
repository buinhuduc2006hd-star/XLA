"""Audit chuyên sâu Giai đoạn 3: Kiểm tra Corner Accuracy, Orientation và ROI alignment.

Script kiểm tra:
1. Chạy YOLOv8n trên tập test (hoặc subset).
2. Chạy DocumentRectifier để lấy 4 góc và ảnh rectified 856x540.
3. Phân tích phân bố góc xoay, tỷ lệ khung hình, chất lượng góc.
4. Kiểm tra orientation thực tế của các ảnh trong tập test.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.document_detector import DocumentDetector
from services.document_rectifier import DocumentRectifier, RectifyResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def audit_corners_and_geometry():
    test_dir = PROJECT_ROOT / "datasets" / "images" / "test"
    image_paths = sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.png"))
    logger.info(f"Tìm thấy {len(image_paths)} ảnh test.")

    model_path = PROJECT_ROOT / "models" / "yolov8n_document.pt"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "runs" / "detect" / "smoke_test_10e" / "weights" / "best.pt"

    detector = DocumentDetector(model_path=str(model_path), conf_threshold=0.25)
    rectifier = DocumentRectifier(margin=0.08)

    stats = {
        "total": len(image_paths),
        "detected_yolo": 0,
        "cccd_count": 0,
        "gplx_count": 0,
        "rectify_success": 0,
        "rectify_fallback": 0,
        "methods": {},
        "aspect_ratios": [],
        "rotation_degs": [],
        "confidences": [],
        "details": [],
    }

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        res = detector.detect(img)
        if not res.has_detection:
            stats["details"].append({
                "file": img_path.name,
                "yolo_detected": False,
            })
            continue

        stats["detected_yolo"] += 1
        det = res.top_detection
        card_type = det.class_name
        if card_type == "cccd":
            stats["cccd_count"] += 1
        else:
            stats["gplx_count"] += 1

        rect_res = rectifier.rectify(img, det.bbox_xyxy, card_type=card_type)
        if rect_res.success and not rect_res.fallback_used:
            stats["rectify_success"] += 1
        else:
            stats["rectify_fallback"] += 1

        stats["methods"][rect_res.method] = stats["methods"].get(rect_res.method, 0) + 1
        stats["aspect_ratios"].append(rect_res.aspect_ratio)
        stats["rotation_degs"].append(rect_res.rotation_deg)
        stats["confidences"].append(rect_res.confidence)

        stats["details"].append({
            "file": img_path.name,
            "yolo_detected": True,
            "card_type": card_type,
            "yolo_conf": round(det.confidence, 4),
            "bbox": det.bbox_xyxy,
            "rect_method": rect_res.method,
            "rect_conf": round(rect_res.confidence, 4),
            "fallback": rect_res.fallback_used,
            "aspect_ratio": round(rect_res.aspect_ratio, 3),
            "rotation_deg": round(rect_res.rotation_deg, 2),
            "corners": rect_res.corners,
        })

    logger.info(f"YOLO detection rate: {stats['detected_yolo']}/{stats['total']} ({stats['detected_yolo']/stats['total']*100:.1f}%)")
    logger.info(f"Rectify success rate: {stats['rectify_success']}/{stats['detected_yolo']} ({stats['rectify_success']/stats['detected_yolo']*100:.1f}%)")
    logger.info(f"Methods: {stats['methods']}")
    logger.info(f"Aspect ratio range: min={min(stats['aspect_ratios']):.2f}, mean={np.mean(stats['aspect_ratios']):.2f}, max={max(stats['aspect_ratios']):.2f}")
    logger.info(f"Rotation deg range: min={min(stats['rotation_degs']):.2f}, mean={np.mean(stats['rotation_degs']):.2f}, max={max(stats['rotation_degs']):.2f}")

    out_file = PROJECT_ROOT / "debug_output" / "rectification" / "audit_stage3_geometry.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Báo cáo lưu tại: {out_file}")


if __name__ == "__main__":
    audit_corners_and_geometry()
