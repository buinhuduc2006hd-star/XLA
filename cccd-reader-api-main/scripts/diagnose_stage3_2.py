"""Diagnose OCR text & coordinates on rectified samples."""
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import easyocr

logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
reader = easyocr.Reader(["vi", "en"], gpu=False)

def inspect_image(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Cannot load {img_path}")
        return []
    h, w = img.shape[:2]
    print(f"\n=======================================================")
    print(f"Inspecting: {img_path.name} (shape: {w}x{h})")
    
    results = reader.readtext(img, mag_ratio=1.5)
    extracted = []
    for box, text, conf in results:
        xs = [pt[0] / w for pt in box]
        ys = [pt[1] / h for pt in box]
        item = {
            "bbox_norm": [round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)],
            "text": text,
            "conf": round(float(conf), 3)
        }
        extracted.append(item)
        print(f"{item['bbox_norm']} conf={item['conf']} : '{text}'")
    return extracted

if __name__ == "__main__":
    dir_path = PROJECT_ROOT / "debug_output" / "stage3_1"
    out_dir = PROJECT_ROOT / "debug_output" / "stage3_2"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(dir_path.glob("*_final_d.jpg"))
    print(f"Found {len(files)} final_d files in {dir_path}")
    all_res = {}
    for f in files:
        all_res[f.name] = inspect_image(f)
        
    with open(out_dir / "diagnosis_raw.json", "w", encoding="utf-8") as out_f:
        json.dump(all_res, out_f, indent=2, ensure_ascii=False)
    print("Saved diagnosis_raw.json")
