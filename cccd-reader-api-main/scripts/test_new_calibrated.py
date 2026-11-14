import json
from pathlib import Path
import cv2
from services.calibrated_rois import (
    extract_calibrated_cccd_fields,
    extract_calibrated_gplx_fields,
)

dir_path = Path("cccd-reader-api-main/debug_output/stage3_1")
samples = [
    ("01_cccd_1794065827572_499_final_d.jpg", "cccd"),
    ("02_cccd_1794065827726_499_final_d.jpg", "cccd"),
    ("03_cccd_1794065837638_499_final_d.jpg", "cccd"),
    ("04_gplx_1794202637314_499_final_d.jpg", "gplx"),
    ("05_gplx_1794202637456_499_final_d.jpg", "gplx"),
    ("06_gplx_1794202637809_499_final_d.jpg", "gplx"),
]

for fname, ctype in samples:
    p = dir_path / fname
    img = cv2.imread(str(p))
    if img is None:
        print(f"Cannot load {p}")
        continue
    if ctype == "cccd":
        res = extract_calibrated_cccd_fields(img)
    else:
        res = extract_calibrated_gplx_fields(img)
    valid = sum(1 for v in res.values() if v is not None)
    tot = len(res)
    print(f"\n[{fname}] -> {valid}/{tot} fields")
    for k, v in res.items():
        print(f"  {k}: {v}")

