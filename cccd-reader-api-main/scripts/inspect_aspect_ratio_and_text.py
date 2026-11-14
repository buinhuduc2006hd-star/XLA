"""Kiểm tra chi tiết các ảnh có aspect ratio bất thường và ảnh xoay 90/180/270 độ."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.document_detector import DocumentDetector
from services.document_rectifier import DocumentRectifier
from services.ocr_service import read_text

def inspect():
    debug_dir = PROJECT_ROOT / "debug_output" / "roi_calibration"
    debug_dir.mkdir(parents=True, exist_ok=True)

    model_path = PROJECT_ROOT / "models" / "yolov8n_document.pt"
    detector = DocumentDetector(model_path=str(model_path), conf_threshold=0.25)
    rectifier = DocumentRectifier(margin=0.08)

    test_dir = PROJECT_ROOT / "datasets" / "images" / "test"
    target_files = [
        "gplx_1794203009931_499751437511627297_2033087232165695992_7724252e44601ccb362dcd690ade5cd6_jpg.rf.f64cbd7aa4576454ade110d04c270096.jpg",
        "cccd_1794065837638_499751437511627297_2033087232165695992_342a3ea81aab8a3f5fabb9624f567fa7_jpg.rf.ac3683d16adc1e56739ddefd1c6436aa.jpg",
        "cccd_1794065923328_499751437511627297_2033087232165695992_0af780318a5e3ef6c534e51dd5eb2ac1_jpg.rf.6b01ca1889526a44ad8ec5cd7d64c91a.jpg",
        "gplx_1794202950249_499751437511627297_2033087232165695992_a1d376a0bc2b47d51caf159f7c1d4ebb_jpg.rf.de9e18f35f23f94b92ebea848638d623.jpg",
    ]

    for fname in target_files:
        p = test_dir / fname
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        res = detector.detect(img)
        if not res.has_detection:
            continue
        det = res.top_detection
        rect_res = rectifier.rectify(img, det.bbox_xyxy, card_type=det.class_name)

        # Save crops and warped
        x1, y1, x2, y2 = det.bbox_xyxy
        yolo_crop = img[y1:y2, x1:x2]

        prefix = fname.split("_")[0] + "_" + fname[:20]
        cv2.imwrite(str(debug_dir / f"{prefix}_orig.jpg"), img)
        cv2.imwrite(str(debug_dir / f"{prefix}_yolo_crop.jpg"), yolo_crop)
        cv2.imwrite(str(debug_dir / f"{prefix}_warped.jpg"), rect_res.rectified_image)

        print(f"\n--- {fname[:35]} ---")
        print(f"Card type: {det.class_name}, BBox: {det.bbox_xyxy}")
        print(f"Rectify: conf={rect_res.confidence:.3f}, aspect={rect_res.aspect_ratio:.3f}, rot={rect_res.rotation_deg:.1f}°")
        print(f"Corners: {rect_res.corners}")

if __name__ == "__main__":
    inspect()

