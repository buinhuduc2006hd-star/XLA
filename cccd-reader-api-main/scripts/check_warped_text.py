import cv2
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.ocr_service import read_text

debug_dir = Path("debug_output/roi_calibration")
warped_files = sorted(debug_dir.glob("*_warped.jpg"))

for wf in warped_files:
    img = cv2.imread(str(wf))
    results = read_text(img)
    texts = [f"{r['text']} (conf={r['confidence']:.2f}, y={r['box'][0][1]})" for r in results if r["confidence"] > 0.3]
    print(f"\n=== {wf.name} ===")
    for t in texts:
        print(" ", t)
