"""Bộ tọa độ ROI hiệu chuẩn chuyên biệt cho ảnh đã qua Rectification (856x540).

Giai đoạn 3.1 & 3.2: ROI Calibration & OCR Accuracy Optimization cho CCCD và GPLX.
- Giữ nguyên bộ ROI gốc trong services/cccd_extractor.py và services/gplx_extractor.py
  để bảo vệ tuyệt đối baseline của Giai đoạn 1.
- Hiệu chuẩn độc lập RECTIFIED_CCCD_ROIS và RECTIFIED_GPLX_ROIS dựa trên số liệu
  đo đạc phân bố thực tế của các khối văn bản trên ảnh chuẩn hóa 856x540 px.
- Khắc phục triệt để:
  + CCCD date_of_birth & full_name regressions.
  + GPLX license_class dot-matrix letter parsing ('Al' -> 'A1').
  + GPLX expiry ("Không thời hạn" & coordinate shift y: 0.74-0.86).
  + GPLX license_number red font recognition.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Callable

import cv2
import numpy as np

from services.cccd_extractor import (
    BOX_PADDING,
    MIN_CONFIDENCE,
    Box,
    _crop_region,
    _normalize_width,
    _ocr_region,
    _tokens,
    _parse_date_string,
    _clean_id_candidate,
    PREPROCESS_VARIANTS,
    _run_ocr,
)
from services.gplx_extractor import (
    _CLASS_DIGIT_FIXES,
    _NUMBER_DIGIT_FIXES,
    VALID_CLASSES,
    DATE_PATTERN,
    _normalize_class,
)

logger = logging.getLogger(__name__)

# ===========================================================================
# 1. BẢNG ROI HIỆU CHUẨN CHO THẺ CCCD (GẮN CHIP 2021) TRÊN ẢNH 856x540
# ===========================================================================

# Tọa độ chuẩn hóa (x1, y1, x2, y2) trên không gian ảnh nắn phẳng [0, 1]
RECTIFIED_BOX_CCCD_ID_NUMBER: Box = (0.37, 0.36, 0.88, 0.51)
RECTIFIED_BOX_CCCD_FULL_NAME: Box = (0.28, 0.49, 0.72, 0.59)  # y2 thu gọn về 0.59 tránh lấn dòng ngày sinh
RECTIFIED_BOX_CCCD_DATE_OF_BIRTH: Box = (0.52, 0.59, 0.88, 0.71)  # mở rộng x1 sang 0.52
RECTIFIED_BOX_CCCD_GENDER: Box = (0.44, 0.63, 0.62, 0.73)
RECTIFIED_BOX_CCCD_NATIONALITY: Box = (0.78, 0.63, 0.99, 0.74)
RECTIFIED_BOX_CCCD_PLACE_OF_ORIGIN: Box = (0.28, 0.70, 0.92, 0.83)
RECTIFIED_BOX_CCCD_RESIDENCE_LINE_1: Box = (0.42, 0.80, 0.98, 0.90)
RECTIFIED_BOX_CCCD_RESIDENCE_LINE_2: Box = (0.28, 0.86, 0.95, 0.97)
RECTIFIED_BOX_CCCD_EXPIRY_DATE: Box = (0.02, 0.74, 0.38, 0.94)

RECTIFIED_CCCD_ROIS: dict[str, Box] = {
    "id_number": RECTIFIED_BOX_CCCD_ID_NUMBER,
    "full_name": RECTIFIED_BOX_CCCD_FULL_NAME,
    "date_of_birth": RECTIFIED_BOX_CCCD_DATE_OF_BIRTH,
    "gender": RECTIFIED_BOX_CCCD_GENDER,
    "nationality": RECTIFIED_BOX_CCCD_NATIONALITY,
    "place_of_origin": RECTIFIED_BOX_CCCD_PLACE_OF_ORIGIN,
    "residence_line_1": RECTIFIED_BOX_CCCD_RESIDENCE_LINE_1,
    "residence_line_2": RECTIFIED_BOX_CCCD_RESIDENCE_LINE_2,
    "expiry_date": RECTIFIED_BOX_CCCD_EXPIRY_DATE,
}

# ===========================================================================
# 2. BẢNG ROI HIỆU CHUẨN CHO THẺ GPLX (PET) TRÊN ẢNH 856x540
# ===========================================================================

# Tọa độ đo đạc thực tế chính xác trên ảnh 856x540
RECTIFIED_BOX_GPLX_LICENSE_NUMBER: Box = (0.48, 0.28, 0.88, 0.42)
RECTIFIED_BOX_GPLX_FULL_NAME: Box = (0.52, 0.35, 0.88, 0.46)
RECTIFIED_BOX_GPLX_DATE_OF_BIRTH: Box = (0.58, 0.40, 0.86, 0.50)
RECTIFIED_BOX_GPLX_NATIONALITY: Box = (0.60, 0.47, 0.86, 0.56)
RECTIFIED_BOX_GPLX_ADDRESS: Box = (0.35, 0.53, 0.95, 0.73)
RECTIFIED_BOX_GPLX_CLASS: Box = (0.12, 0.67, 0.32, 0.78)  # Chuẩn hóa vùng Hạng bằng
RECTIFIED_BOX_GPLX_EXPIRES: Box = (0.20, 0.74, 0.65, 0.86)  # Hạ tọa độ y bắt đúng dòng Ngày hết hạn

RECTIFIED_GPLX_ROIS: dict[str, Box] = {
    "license_number": RECTIFIED_BOX_GPLX_LICENSE_NUMBER,
    "full_name": RECTIFIED_BOX_GPLX_FULL_NAME,
    "date_of_birth": RECTIFIED_BOX_GPLX_DATE_OF_BIRTH,
    "nationality": RECTIFIED_BOX_GPLX_NATIONALITY,
    "address": RECTIFIED_BOX_GPLX_ADDRESS,
    "license_class": RECTIFIED_BOX_GPLX_CLASS,
    "expiry": RECTIFIED_BOX_GPLX_EXPIRES,
}


# ===========================================================================
# TIỀN XỬ LÝ NÂNG CAO CHO CÁC TRƯỜNG ĐẶC THÙ
# ===========================================================================

def _preprocess_green_channel(crop: np.ndarray) -> np.ndarray:
    """Tách kênh Green để tối đa tương phản chữ đỏ (Số GPLX thẻ PET)."""
    if len(crop.shape) == 3:
        green = crop[:, :, 1]
    else:
        green = crop
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)


def _preprocess_dot_matrix(crop: np.ndarray) -> np.ndarray:
    """Tiền xử lý tăng cường cho chữ ma trận chấm / in kim (Hạng GPLX A1/B2)."""
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    # Nhị phân hóa Otsu để tôn nét chữ
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphology đóng nhẹ để nối nét chấm
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return closed


def _ocr_region_smart(
    image: np.ndarray,
    field: str,
    box: Box,
    validator: Callable[[str], bool] | None = None,
    min_confidence: float = 0.25,
    custom_preprocessor: Callable[[np.ndarray], np.ndarray] | None = None,
    fast_mode: bool = False,
) -> list[str]:
    """OCR vùng ROI với Early Exit khi kết quả thỏa mãn validator, hỗ trợ profiling và bảo vệ PII."""
    t_start = time.perf_counter()
    crop = _crop_region(image, field, box)
    h_c, w_c = crop.shape[:2]

    variants = list(PREPROCESS_VARIANTS)
    if custom_preprocessor is not None:
        variants.insert(0, ("custom_smart", custom_preprocessor))

    if fast_mode:
        # Fast mode: chỉ chạy biến thể ưu tiên để tối ưu tốc độ CPU
        variants = [variants[0]]

    best_parts: list[str] = []
    best_weight: float = -1.0
    ocr_calls = 0
    total_prep_ms = 0.0
    total_ocr_ms = 0.0
    early_exit_used = False

    for name, prep in variants:
        t_p0 = time.perf_counter()
        try:
            prepped = prep(crop)
        except Exception as e:
            logger.debug(f"Preprocess variant {name} failed: {type(e).__name__}")
            continue
        total_prep_ms += (time.perf_counter() - t_p0) * 1000.0

        t_o0 = time.perf_counter()
        try:
            cand = _run_ocr(prepped, field, name, min_confidence=min_confidence)
            ocr_calls += 1
        except Exception as e:
            logger.debug(f"OCR variant {name} failed: {type(e).__name__}")
            continue
        total_ocr_ms += (time.perf_counter() - t_o0) * 1000.0

        if cand.parts:
            # Nếu có validator và bất kỳ phần nào thỏa mãn -> Early Exit ngay lập tức!
            if validator:
                for p in cand.parts:
                    if validator(p):
                        early_exit_used = True
                        logger.debug(f"{field} Early Exit via validator on variant '{name}' (len={len(p)})")
                        best_parts = cand.parts
                        break
                if early_exit_used:
                    break

                joined = " ".join(cand.parts)
                if validator(joined):
                    early_exit_used = True
                    logger.debug(f"{field} Early Exit via validator (joined) on variant '{name}' (len={len(joined)})")
                    best_parts = cand.parts
                    break

            if fast_mode:
                best_parts = cand.parts
                break

            if cand.weight > best_weight:
                best_weight = cand.weight
                best_parts = cand.parts

    total_time_ms = (time.perf_counter() - t_start) * 1000.0
    # Profiling log không chứa bất kỳ nội dung OCR nhạy cảm nào (PII-safe):
    logger.debug(
        f"ROI Profile [{field}]: dims={w_c}x{h_c}, ocr_calls={ocr_calls}, "
        f"prep_ms={total_prep_ms:.1f}, ocr_ms={total_ocr_ms:.1f}, "
        f"early_exit={early_exit_used}, retries={max(0, ocr_calls - 1)}, total_ms={total_time_ms:.1f}"
    )

    return best_parts


# ===========================================================================
# HÀM TRÍCH XUẤT CHO CCCD & GPLX
# ===========================================================================

def extract_calibrated_cccd_fields(
    image: np.ndarray,
    fast_mode: bool = False,
) -> dict[str, str | None]:
    """Trích xuất trường CCCD sử dụng bộ ROI đã hiệu chuẩn và smart OCR."""
    norm_image = _normalize_width(image)

    # 1. Số CCCD: 12 chữ số
    def _is_valid_id(val: str) -> bool:
        c = _clean_id_candidate(val)
        return bool(c and len(c) == 12)

    id_parts = _ocr_region_smart(
        norm_image,
        "id_number",
        RECTIFIED_BOX_CCCD_ID_NUMBER,
        validator=_is_valid_id,
        fast_mode=fast_mode,
    )
    id_number = None
    for part in id_parts:
        c = _clean_id_candidate(part)
        if c:
            id_number = c
            break
    if not id_number and id_parts:
        id_number = _clean_id_candidate("".join(id_parts))

    # 2. Họ và tên: làm sạch các nhãn có thể lẫn
    name_parts = _ocr_region_smart(
        norm_image,
        "full_name",
        RECTIFIED_BOX_CCCD_FULL_NAME,
        fast_mode=fast_mode,
    )
    full_name = None
    if name_parts:
        cleaned_parts = []
        noise_labels = [
            "Họ và tên", "Họ tên", "Full name", "TEN", "FULL", "NAME", "HO", "VA",
            "Ngày sinh", "Date of birth", "sinh", "Ngay", "birth", "Dale",
        ]
        for p in name_parts:
            p_clean = p
            for lbl in noise_labels:
                p_clean = re.sub(rf"(?i)\b{re.escape(lbl)}\b", "", p_clean)
            p_clean = p_clean.strip()
            # Tên người phải không có chữ số và độ dài tối thiểu 2 ký tự
            if p_clean and len(p_clean) >= 2 and not any(ch.isdigit() for ch in p_clean):
                cleaned_parts.append(p_clean)
        if cleaned_parts:
            full_name = " ".join(cleaned_parts).upper()

    # 3. Ngày sinh
    def _is_valid_date(val: str) -> bool:
        return _parse_date_string(val) is not None

    dob_parts = _ocr_region_smart(
        norm_image,
        "date_of_birth",
        RECTIFIED_BOX_CCCD_DATE_OF_BIRTH,
        validator=_is_valid_date,
        fast_mode=fast_mode,
    )
    date_of_birth = None
    for p in dob_parts:
        d = _parse_date_string(p)
        if d:
            date_of_birth = d
            break
    if not date_of_birth and dob_parts:
        # Thử ghép toàn bộ chuỗi
        date_of_birth = _parse_date_string(" ".join(dob_parts))

    # 4. Giới tính
    gender_parts = _ocr_region_smart(
        norm_image,
        "gender",
        RECTIFIED_BOX_CCCD_GENDER,
        fast_mode=fast_mode,
    )
    gender = None
    for p in gender_parts:
        p_up = p.upper()
        if "NAM" in p_up and "VIET" not in p_up:
            gender = "Nam"
            break
        elif "NU" in p_up or "NỮ" in p_up:
            gender = "Nữ"
            break

    # 5. Quốc tịch
    nat_parts = _ocr_region_smart(
        norm_image,
        "nationality",
        RECTIFIED_BOX_CCCD_NATIONALITY,
        fast_mode=fast_mode,
    )
    nationality = "Việt Nam" if any("VIET" in p.upper() or "NAM" in p.upper() for p in nat_parts) else None

    # 6. Quê quán
    orig_parts = _ocr_region_smart(
        norm_image,
        "place_of_origin",
        RECTIFIED_BOX_CCCD_PLACE_OF_ORIGIN,
        fast_mode=fast_mode,
    )
    place_of_origin = " ".join(orig_parts) if orig_parts else None

    # 7. Nơi thường trú
    res1_parts = _ocr_region_smart(
        norm_image,
        "residence_line_1",
        RECTIFIED_BOX_CCCD_RESIDENCE_LINE_1,
        fast_mode=fast_mode,
    )
    res2_parts = _ocr_region_smart(
        norm_image,
        "residence_line_2",
        RECTIFIED_BOX_CCCD_RESIDENCE_LINE_2,
        fast_mode=fast_mode,
    )
    res_lines = [" ".join(res1_parts), " ".join(res2_parts)]
    residence = ", ".join([l for l in res_lines if l.strip()]) or None

    # 8. Ngày hết hạn
    exp_parts = _ocr_region_smart(
        norm_image,
        "expiry_date",
        RECTIFIED_BOX_CCCD_EXPIRY_DATE,
        validator=_is_valid_date,
        fast_mode=fast_mode,
    )
    expiry_date = None
    for p in exp_parts:
        if "KHONG" in p.upper() or "THOI HAN" in p.upper() or "KHÔNG" in p.upper():
            expiry_date = "Không thời hạn"
            break
        d = _parse_date_string(p)
        if d:
            expiry_date = d
            break
    if not expiry_date and exp_parts:
        expiry_date = _parse_date_string(" ".join(exp_parts))

    return {
        "id_number": id_number,
        "full_name": full_name,
        "date_of_birth": date_of_birth,
        "gender": gender,
        "nationality": nationality,
        "place_of_origin": place_of_origin,
        "residence": residence,
        "expiry_date": expiry_date,
    }


def extract_calibrated_gplx_fields(
    image: np.ndarray,
    fast_mode: bool = False,
) -> dict[str, str | None]:
    """Trích xuất trường GPLX sử dụng bộ ROI đã hiệu chuẩn và smart OCR."""
    norm_image = _normalize_width(image)

    # 1. Số GPLX: dùng kênh Green để tương phản tối đa chữ đỏ
    def _is_valid_lic(val: str) -> bool:
        digits = re.sub(r"\D", "", val.translate(_NUMBER_DIGIT_FIXES))
        return bool(re.search(r"\d{12}", digits) or re.search(r"\d{9,12}", digits))

    lic_parts = _ocr_region_smart(
        norm_image,
        "license_number",
        RECTIFIED_BOX_GPLX_LICENSE_NUMBER,
        validator=_is_valid_lic,
        custom_preprocessor=_preprocess_green_channel,
        fast_mode=fast_mode,
    )
    text = " ".join(lic_parts)
    digits = re.sub(r"\D", "", text.translate(_NUMBER_DIGIT_FIXES))
    match = re.search(r"\d{12}", digits) or re.search(r"\d{9,12}", digits)
    license_number = match.group(0) if match else None

    # 2. Họ và tên
    name_parts = _ocr_region_smart(
        norm_image,
        "full_name",
        RECTIFIED_BOX_GPLX_FULL_NAME,
        fast_mode=fast_mode,
    )
    cleaned_name_parts = []
    noise_labels = ["Họ tên", "Full name", "HO", "TEN", "FULL", "NAME", "Số", "No"]
    for p in name_parts:
        p_clean = p
        for lbl in noise_labels:
            p_clean = re.sub(rf"(?i)\b{re.escape(lbl)}\b", "", p_clean)
        p_clean = p_clean.strip()
        if p_clean and len(p_clean) >= 2 and not any(ch.isdigit() for ch in p_clean):
            cleaned_name_parts.append(p_clean)
    full_name = " ".join(cleaned_name_parts).upper() if cleaned_name_parts else None

    # 3. Ngày sinh
    def _is_valid_dob(val: str) -> bool:
        return DATE_PATTERN.search(val) is not None or _parse_date_string(val) is not None

    dob_parts = _ocr_region_smart(
        norm_image,
        "date_of_birth",
        RECTIFIED_BOX_GPLX_DATE_OF_BIRTH,
        validator=_is_valid_dob,
        fast_mode=fast_mode,
    )
    date_of_birth = None
    joined_dob = " ".join(dob_parts)
    m = DATE_PATTERN.search(joined_dob)
    if m:
        date_of_birth = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    else:
        date_of_birth = _parse_date_string(joined_dob)

    # 4. Quốc tịch
    nat_parts = _ocr_region_smart(
        norm_image,
        "nationality",
        RECTIFIED_BOX_GPLX_NATIONALITY,
        fast_mode=fast_mode,
    )
    nationality = "Việt Nam" if any("VIET" in p.upper() or "NAM" in p.upper() for p in nat_parts) else None

    # 5. Địa chỉ
    addr_parts = _ocr_region_smart(
        norm_image,
        "address",
        RECTIFIED_BOX_GPLX_ADDRESS,
        fast_mode=fast_mode,
    )
    address = " ".join(addr_parts) if addr_parts else None

    # 6. Hạng bằng (tiền xử lý dot-matrix + chuẩn hóa token Al -> A1)
    def _is_valid_class(val: str) -> bool:
        return _normalize_class(val) is not None

    class_parts = _ocr_region_smart(
        norm_image,
        "license_class",
        RECTIFIED_BOX_GPLX_CLASS,
        validator=_is_valid_class,
        min_confidence=0.15,  # Hạ ngưỡng để bắt ký tự in mờ
        custom_preprocessor=_preprocess_dot_matrix,
        fast_mode=fast_mode,
    )
    license_class = None
    for p in class_parts:
        c = _normalize_class(p)
        if c:
            license_class = c
            break
    if not license_class and class_parts:
        license_class = _normalize_class(" ".join(class_parts))

    # 7. Ngày hết hạn: Hỗ trợ "Không thời hạn" và ngày cụ thể
    def _is_valid_exp(val: str) -> bool:
        toks = _tokens(val)
        return bool(toks & {"khong", "thoi", "han"}) or DATE_PATTERN.search(val) is not None

    exp_parts = _ocr_region_smart(
        norm_image,
        "expiry",
        RECTIFIED_BOX_GPLX_EXPIRES,
        validator=_is_valid_exp,
        min_confidence=0.15,
        fast_mode=fast_mode,
    )
    expiry = None
    joined_exp = " ".join(exp_parts)
    tokens_exp = _tokens(joined_exp)
    if tokens_exp & {"khong", "thoi", "han"}:
        expiry = "Không thời hạn"
    else:
        m = DATE_PATTERN.search(joined_exp)
        if m:
            expiry = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
        else:
            expiry = _parse_date_string(joined_exp)

    return {
        "license_number": license_number,
        "full_name": full_name,
        "date_of_birth": date_of_birth,
        "nationality": nationality,
        "address": address,
        "license_class": license_class,
        "expiry": expiry,
    }
