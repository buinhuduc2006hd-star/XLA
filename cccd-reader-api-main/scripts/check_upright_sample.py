import cv2
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.ocr_service import read_text

debug_dir = Path("debug_output/rectification")
warped_sample = list(debug_dir.glob("rectified_warp_cccd_1794065827572*.jpg"))
if warped_sample:
    img = cv2.imread(str(warped_sample[0]))
    results = read_text(img)
    print(f"=== {warped_sample[0].name[:40]} ===")
    for r in results:
        if r["confidence"] > 0.3:
            print(f"  {r['text']} (conf={r['confidence']:.2f}, y={r['box'][0][1]}, x={r['box'][0][0]})")

