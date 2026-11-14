from pathlib import Path
import sys
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.document_detector import DocumentDetector
from services.document_rectifier import DocumentRectifier
from services.document_orientation import correct_orientation
from services.calibrated_rois import extract_calibrated_cccd_fields, extract_calibrated_gplx_fields
from services.cccd_extractor import (
    _normalize_width,
    extract_id_number,
    extract_full_name,
    extract_date_of_birth,
    extract_gender,
    extract_nationality,
    extract_place_of_origin,
    extract_residence,
    extract_expiry_date,
)
from services.gplx_extractor import (
    extract_license_number,
    extract_full_name as extract_gplx_name,
    extract_date_of_birth as extract_gplx_dob,
    extract_nationality as extract_gplx_nat,
    extract_address,
    extract_license_class,
    extract_expiry,
)

def extract_cccd_old(img):
    norm = _normalize_width(img)
    return {
        "id_number": extract_id_number(norm),
        "full_name": extract_full_name(norm),
        "date_of_birth": extract_date_of_birth(norm),
        "gender": extract_gender(norm),
        "nationality": extract_nationality(norm),
        "place_of_origin": extract_place_of_origin(norm),
        "residence": extract_residence(norm),
        "expiry_date": extract_expiry_date(norm),
    }

def extract_gplx_old(img):
    norm = _normalize_width(img)
    return {
        "license_number": extract_license_number(norm),
        "full_name": extract_gplx_name(norm),
        "date_of_birth": extract_gplx_dob(norm),
        "nationality": extract_gplx_nat(norm),
        "address": extract_address(norm),
        "license_class": extract_license_class(norm),
        "expiry": extract_expiry(norm),
    }

model_path = PROJECT_ROOT / "models" / "yolov8n_document.pt"
detector = DocumentDetector(model_path=str(model_path), conf_threshold=0.25)
rectifier = DocumentRectifier(margin=0.08)

test_dir = PROJECT_ROOT / "datasets" / "images" / "test"
samples = [
    ("cccd_1794065827572_499751437511627297_2033087232165695992_ffd2de2350e51f61744ef1ff9c530a1d_jpg.rf.5346da3778765ca825ef1a89d15591fc.jpg", "cccd"),
    ("gplx_1794202637809_499751437511627297_2033087232165695992_c2be8410a950dc8270f96203aadbf374_jpg.rf.32932bc2a4ed8aef21d15189818eb64a.jpg", "gplx"),
]

for fname, card_type in samples:
    p = test_dir / fname
    img = cv2.imread(str(p))
    det_res = detector.detect(img)
    det = det_res.top_detection
    rect_res = rectifier.rectify(img, det.bbox_xyxy, card_type=card_type)
    orient_res = correct_orientation(rect_res.rectified_image, card_type=card_type)
    warped = orient_res.image

    print(f"\n==================== {fname[:35]} ({card_type.upper()}) ====================")
    if card_type == "cccd":
        old_res = extract_cccd_old(warped)
        new_res = extract_calibrated_cccd_fields(warped)
    else:
        old_res = extract_gplx_old(warped)
        new_res = extract_calibrated_gplx_fields(warped)

    print("OLD (Stage 1 ROIs):")
    for k, v in old_res.items():
        print(f"  {k:15}: {v}")

    print("\nNEW (Calibrated ROIs):")
    for k, v in new_res.items():
        print(f"  {k:15}: {v}")
