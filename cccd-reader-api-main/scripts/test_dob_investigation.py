import cv2
from pathlib import Path
from services.cccd_extractor import _normalize_width, _crop_region, _ocr_region, _parse_date_string, PREPROCESS_VARIANTS, _run_ocr
from services.calibrated_rois import RECTIFIED_BOX_CCCD_DATE_OF_BIRTH, RECTIFIED_BOX_CCCD_FULL_NAME

img_path = Path("cccd-reader-api-main/debug_output/stage3_1/01_cccd_1794065827572_499_final_d.jpg")
img = cv2.imread(str(img_path))
norm = _normalize_width(img)

print("--- Testing DOB on 01_cccd ---")
crop = _crop_region(norm, "date_of_birth", RECTIFIED_BOX_CCCD_DATE_OF_BIRTH)
print(f"Crop shape: {crop.shape}")

for name, prep in PREPROCESS_VARIANTS:
    cand = _run_ocr(prep(crop), "date_of_birth", name, min_confidence=0.1)
    print(f"Variant '{name}': parts={cand.parts}, conf={cand.mean_confidence:.2f}")
    for p in cand.parts:
        print(f"  _parse_date_string('{p}') -> {_parse_date_string(p)}")

print("\n--- Testing Full Name on 01_cccd ---")
name_crop = _crop_region(norm, "full_name", RECTIFIED_BOX_CCCD_FULL_NAME)
for name, prep in PREPROCESS_VARIANTS:
    cand = _run_ocr(prep(name_crop), "full_name", name, min_confidence=0.1)
    print(f"Variant '{name}': parts={cand.parts}")

