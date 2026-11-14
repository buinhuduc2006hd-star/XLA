"""Calibrate ROIs for rectified 856x540 CCCD and GPLX cards.

Script maps text positions across sample rectified images to find optimal
calibrated ROIs for Pipeline D.
"""

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
from services.document_orientation import correct_orientation
from services.ocr_service import read_text
from services.cccd_extractor import CCCD_ROIS
from services.gplx_extractor import GPLX_ROIS

debug_dir = PROJECT_ROOT / "debug_output" / "roi_calibration"
debug_dir.mkdir(parents=True, exist_ok=True)


def analyze_card(img_path: Path, detector, rectifier):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    det_res = detector.detect(img)
    if not det_res.has_detection:
        return None

    det = det_res.top_detection
    card_type = det.class_name

    rect_res = rectifier.rectify(img, det.bbox_xyxy, card_type=card_type)
    if not rect_res.success:
        return None

    orient_res = correct_orientation(rect_res.rectified_image, card_type=card_type)
    rect_img = orient_res.image  # shape is (540, 856, 3)

    ocr_results = read_text(rect_img)
    h, w = rect_img.shape[:2]

    # Save debug image
    annotated = rect_img.copy()

    # Log text tokens with normalized coords
    tokens = []
    for r in ocr_results:
        box = r["box"]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1_norm = round(min(xs) / w, 3)
        y1_norm = round(min(ys) / h, 3)
        x2_norm = round(max(xs) / w, 3)
        y2_norm = round(max(ys) / h, 3)

        tokens.append({
            "text": r["text"],
            "conf": round(r["confidence"], 3),
            "norm_box": [x1_norm, y1_norm, x2_norm, y2_norm],
            "pixel_box": [min(xs), min(ys), max(xs), max(ys)],
        })

        cv2.rectangle(annotated, (min(xs), min(ys)), (max(xs), max(ys)), (0, 255, 0), 1)
        cv2.putText(annotated, r["text"][:15], (min(xs), max(15, min(ys) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imwrite(str(debug_dir / f"ocr_map_{card_type}_{img_path.stem[:25]}.jpg"), annotated)

    return {
        "file": img_path.name,
        "card_type": card_type,
        "rotation": orient_res.rotation_deg,
        "tokens": tokens,
    }


def main():
    model_path = PROJECT_ROOT / "models" / "yolov8n_document.pt"
    detector = DocumentDetector(model_path=str(model_path), conf_threshold=0.25)
    rectifier = DocumentRectifier(margin=0.08)

    test_dir = PROJECT_ROOT / "datasets" / "images" / "test"
    cccd_files = list(test_dir.glob("cccd_*.jpg"))[:3]
    gplx_files = list(test_dir.glob("gplx_*.jpg"))[:3]

    results = []
    for p in cccd_files + gplx_files:
        print(f"Analyzing {p.name[:35]}...")
        data = analyze_card(p, detector, rectifier)
        if data:
            results.append(data)

    out_file = debug_dir / "roi_calibration_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Analysis saved to {out_file}")


if __name__ == "__main__":
    main()

