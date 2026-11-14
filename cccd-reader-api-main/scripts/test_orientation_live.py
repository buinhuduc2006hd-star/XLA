import time
from pathlib import Path
import sys
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.document_orientation import correct_orientation

debug_dir = PROJECT_ROOT / "debug_output" / "roi_calibration"
samples = [
    (debug_dir / "cccd_cccd_1794065923328_4_warped.jpg", "cccd"),
    (list((PROJECT_ROOT / "debug_output" / "rectification").glob("rectified_warp_cccd_1794065827572*.jpg"))[0], "cccd"),
]

for p, card_type in samples:
    if not p.exists():
        continue
    img = cv2.imread(str(p))
    t0 = time.perf_counter()
    res = correct_orientation(img, card_type=card_type)
    elapsed = time.perf_counter() - t0

    print(f"\nFile: {p.name[:35]}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Success: {res.success}, Rotation: {res.rotation_deg}°, Conf: {res.confidence:.3f}")
    print(f"Details: {res.details}")

