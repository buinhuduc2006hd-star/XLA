"""Trích xuất thông tin từ ảnh GPLX (mẫu thẻ PET có chữ "Nơi cư trú/Address") bằng cách cắt vùng theo tỷ lệ rồi OCR.

Chạy thử / xem OCR thô của từng trường (từ thư mục gốc của project):

    python -m services.gplx_extractor duong_dan_anh.jpg
"""

from __future__ import annotations

import logging
import re

import cv2
import numpy as np

from services.cccd_extractor import (
    DATE_PATTERN,
    Box,
    _normalize_width,
    _ocr_region,
    _tokens,
)

logger = logging.getLogger(__name__)

# (x1, y1, x2, y2) tính theo tỷ lệ; chỉ bao phần GIÁ TRỊ, không lẫn nhãn.
BOX_LICENSE_NUMBER: Box = (0.61, 0.235, 0.86, 0.315)
BOX_FULL_NAME: Box = (0.62, 0.315, 0.85, 0.380)
BOX_DATE_OF_BIRTH: Box = (0.675, 0.385, 0.84, 0.455)
BOX_NATIONALITY: Box = (0.675, 0.455, 0.85, 0.525)
BOX_ADDRESS: Box = (0.38, 0.580, 0.91, 0.665)  # có thể xuống 2 dòng nên để cao hơn 1 dòng
BOX_CLASS: Box = (0.165, 0.795, 0.245, 0.865)
BOX_EXPIRES: Box = (0.28, 0.895, 0.55, 0.965)

VALID_CLASSES: frozenset[str] = frozenset(
    {"A1", "A2", "A3", "A4", "B1", "B2", "C", "D", "E", "FB2", "FC", "FD", "FE"}
)

# OCR hay nhầm chữ <-> số trong hạng bằng lái (A1 -> Al / AI / A|)
_CLASS_DIGIT_FIXES = str.maketrans({"L": "1", "I": "1", "|": "1", "O": "0", "Z": "2"})
_NUMBER_DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8"})


def extract_license_number(image: np.ndarray) -> str | None:
    text = " ".join(_ocr_region(image, "license_number", BOX_LICENSE_NUMBER))
    digits = re.sub(r"\D", "", text.translate(_NUMBER_DIGIT_FIXES))

    # GPLX mới 12 số, thẻ PET cũ có thể 9-12 số
    match = re.search(r"\d{12}", digits) or re.search(r"\d{9,12}", digits)
    return match.group(0) if match else None


def extract_full_name(image: np.ndarray) -> str | None:
    parts = _ocr_region(image, "full_name", BOX_FULL_NAME)
    return " ".join(parts).upper() or None


def extract_date_of_birth(image: np.ndarray) -> str | None:
    parts = _ocr_region(image, "date_of_birth", BOX_DATE_OF_BIRTH)
    if not parts:
        return None

    match = DATE_PATTERN.search(" ".join(parts))
    if match:
        return re.sub(r"\s+", "", match.group(0))
    return parts[0]


def extract_nationality(image: np.ndarray) -> str | None:
    tokens = _tokens(" ".join(_ocr_region(image, "nationality", BOX_NATIONALITY)))

    if "nam" in tokens or any(token.startswith("viet") for token in tokens):
        return "Việt Nam"
    return None


def extract_address(image: np.ndarray) -> str | None:
    parts = _ocr_region(image, "address", BOX_ADDRESS)
    return " ".join(parts) or None


def _normalize_class(raw: str) -> str | None:
    text = re.sub(r"[^A-Za-z0-9|]", "", raw).upper()
    if not text:
        return None

    if text in VALID_CLASSES:
        return text

    # Giữ nguyên chữ cái đầu, chỉ sửa các ký tự sau về dạng số
    fixed = text[0] + text[1:].translate(_CLASS_DIGIT_FIXES)
    if fixed in VALID_CLASSES:
        return fixed

    # Chỉ tìm thấy một hạng hợp lệ nằm trong chuỗi (vd: "HANG A1" -> "A1")
    for candidate in sorted(VALID_CLASSES, key=len, reverse=True):
        if candidate in fixed:
            return candidate
    return None


def extract_license_class(image: np.ndarray) -> str | None:
    parts = _ocr_region(image, "license_class", BOX_CLASS)
    return _normalize_class("".join(parts))


def extract_expiry(image: np.ndarray) -> str | None:
    parts = _ocr_region(image, "expiry", BOX_EXPIRES)
    if not parts:
        return None

    text = " ".join(parts)
    match = DATE_PATTERN.search(text)
    if match:
        return re.sub(r"\s+", "", match.group(0))

    # "Không thời hạn": chấp nhận khi OCR mất dấu hoặc chỉ đọc được một phần
    tokens = _tokens(text)
    if tokens & {"khong", "thoi", "han"}:
        return "Không thời hạn"
    return None


def extract_gplx_fields(image_path: str) -> dict[str, str | None]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = _normalize_width(image)

    return {
        "license_number": extract_license_number(image),
        "full_name": extract_full_name(image),
        "date_of_birth": extract_date_of_birth(image),
        "nationality": extract_nationality(image),
        "address": extract_address(image),
        "license_class": extract_license_class(image),
        "expiry": extract_expiry(image),
    }


if __name__ == "_main_":
    import json
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m services.gplx_extractor <image_path>")

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    print(json.dumps(extract_gplx_fields(sys.argv[1]), ensure_ascii=False, indent=2))