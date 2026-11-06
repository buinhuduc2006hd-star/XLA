"""Trích xuất thông tin từ ảnh CCCD (mặt trước) bằng cách cắt vùng theo tỷ lệ rồi OCR.

Chạy thử / xem OCR thô của từng trường (từ thư mục gốc của project):

    python -m services.cccd_extractor duong_dan_anh.jpg

Hạn chế đã biết
---------------
- Trường ``expiry_date`` ("Có giá trị đến") nằm ở **MẶT SAU** trên CCCD chip mới (2021+).
  Nếu ảnh đầu vào là mặt trước, hàm sẽ trả ``None`` và log WARNING.
  Giải pháp dài hạn: API nhận 2 ảnh riêng (mặt trước + mặt sau).

- Tọa độ ROI của ``id_number`` ước tính theo layout chuẩn Bộ Công an 2021.
  Cần kiểm tra lại với ảnh CCCD thật; các vị trí đánh dấu ``# CALIBRATION NEEDED``.
"""

from __future__ import annotations

import logging
import re
import tempfile
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from services.ocr_service import read_text

logger = logging.getLogger(__name__)

# (x1, y1, x2, y2) tính theo tỷ lệ chiều rộng / chiều cao của ảnh
Box = tuple[float, float, float, float]

# Mỗi vùng chỉ bao đúng phần GIÁ TRỊ của trường,
# không lẫn nhãn ("Giới tính / Sex"...) hay dòng của trường kế bên.
BOX_FULL_NAME: Box = (0.27, 0.53, 0.61, 0.63)
BOX_DATE_OF_BIRTH: Box = (0.56, 0.61, 0.76, 0.70)
BOX_GENDER: Box = (0.46, 0.665, 0.545, 0.74)
BOX_NATIONALITY: Box = (0.81, 0.645, 0.985, 0.735)
BOX_PLACE_OF_ORIGIN: Box = (0.28, 0.79, 0.82, 0.855)
BOX_RESIDENCE_LINE_1: Box = (0.665, 0.855, 0.93, 0.92)  # vd: "Thôn Bích Đông"
BOX_RESIDENCE_LINE_2: Box = (0.28, 0.91, 0.84, 0.985)  # vd: "Nam Chính, Nam Sách, Hải Dương"

# --- Trường mới: Số CCCD (mặt trước) ---
# CALIBRATION NEEDED: tọa độ ước tính theo layout CCCD chip 2021 (Bộ Công an).
# Số 12 chữ số nằm ở góc trên-phải mặt trước, bên dưới quốc huy, trên dòng họ tên.
# Cần kiểm tra lại với ảnh thật và điều chỉnh nếu cần.
BOX_ID_NUMBER: Box = (0.57, 0.42, 0.98, 0.52)  # CALIBRATION NEEDED

# --- Trường mới: Có giá trị đến (MẶT SAU trên CCCD chip 2021+) ---
# CALIBRATION NEEDED: trường này nằm mặt sau, dưới vân tay phải.
# Nếu ảnh đầu vào là mặt trước, OCR khu vực này sẽ không tìm được ngày → trả None.
# Nếu ảnh đầu vào là mặt sau, tọa độ ước tính khu vực chứa "Có giá trị đến / Valid until".
BOX_EXPIRY_DATE: Box = (0.50, 0.75, 0.95, 0.88)  # CALIBRATION NEEDED (mặt sau)

STANDARD_WIDTH = 1061  # chuẩn hoá ảnh về cùng độ rộng để tỷ lệ box ổn định
BOX_PADDING = 0.008  # nới nhẹ mỗi cạnh box (theo tỷ lệ ảnh)
MIN_CONFIDENCE = 0.3  # ảnh chụp thường mờ nên hạ ngưỡng so với 0.5
EARLY_EXIT_CONFIDENCE = 0.75  # đủ tốt thì không cần thử biến thể tiếp theo
UPSCALE = 2.0  # vùng cắt rất nhỏ, phóng to giúp OCR đọc tốt hơn

DATE_PATTERN = re.compile(r"\d{2}\s*[/.\-]\s*\d{2}\s*[/.\-]\s*\d{4}")

# Pattern nhận dạng 12 chữ số CCCD; cho phép ký tự phân cách phổ biến giữa các nhóm
# (OCR đôi khi chèn dấu cách hoặc gạch ngang vào chuỗi số).
_ID_NUMBER_DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8"})
_ID_12_PATTERN = re.compile(r"\d{12}")


@dataclass(frozen=True)
class OcrCandidate:
    variant: str
    parts: list[str]
    mean_confidence: float
    weight: float  # tổng (confidence * số ký tự): ưu tiên biến thể đọc được nhiều chữ tin cậy


# ---------------------------------------------------------------------------
# Tiền xử lý
# ---------------------------------------------------------------------------


def _normalize_width(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if width == STANDARD_WIDTH:
        return image

    scale = STANDARD_WIDTH / width
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)


def _enhance_gray(crop: np.ndarray) -> np.ndarray:
    """Xám -> khử nhiễu -> CLAHE -> làm nét nhẹ (unsharp mask)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)


def _binarize(crop: np.ndarray) -> np.ndarray:
    """Nhị phân hoá thích nghi: hiệu quả khi nền loang màu / sáng không đều."""
    return cv2.adaptiveThreshold(
        _enhance_gray(crop),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        12,
    )


def _identity(crop: np.ndarray) -> np.ndarray:
    return crop


# Thứ tự thử: ảnh xám tăng cường thường tốt nhất cho CCCD chụp, ảnh màu gốc làm dự phòng.
PREPROCESS_VARIANTS: tuple[tuple[str, Callable[[np.ndarray], np.ndarray]], ...] = (
    ("clahe", _enhance_gray),
    ("color", _identity),
    ("binary", _binarize),
)


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def _crop_region(image: np.ndarray, field: str, box: Box) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box

    left = max(int(width * (x1 - BOX_PADDING)), 0)
    right = min(int(width * (x2 + BOX_PADDING)), width)
    top = max(int(height * (y1 - BOX_PADDING)), 0)
    bottom = min(int(height * (y2 + BOX_PADDING)), height)

    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError(f"Invalid {field} crop region")
    return crop


def _run_ocr(processed: np.ndarray, field: str, variant: str) -> OcrCandidate:
    upscaled = cv2.resize(processed, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

    # Dùng thư mục tạm riêng cho mỗi lần gọi: không để rác trong thư mục upload
    # và không bị ghi đè khi có nhiều request chạy cùng lúc.
    with tempfile.TemporaryDirectory() as tmp_dir:
        crop_path = Path(tmp_dir) / "crop.png"
        if not cv2.imwrite(str(crop_path), upscaled):
            raise ValueError(f"Cannot save {field} crop")
        results = read_text(str(crop_path))

    logger.debug(
        "%s [%s] raw OCR: %s",
        field,
        variant,
        [(item["text"], round(float(item["confidence"]), 2)) for item in results],
    )

    kept = [
        (item["text"].strip(), float(item["confidence"]))
        for item in results
        if item["confidence"] >= MIN_CONFIDENCE and re.search(r"\w", item["text"])
    ]
    if not kept:
        return OcrCandidate(variant, [], 0.0, 0.0)

    return OcrCandidate(
        variant=variant,
        parts=[text for text, _ in kept],
        mean_confidence=sum(conf for _, conf in kept) / len(kept),
        weight=sum(conf * len(text) for text, conf in kept),
    )


def _ocr_region(image: np.ndarray, field: str, box: Box) -> list[str]:
    """Cắt vùng theo tỷ lệ, thử nhiều cách tiền xử lý, trả về kết quả OCR tốt nhất."""
    crop = _crop_region(image, field, box)

    best: OcrCandidate | None = None
    for name, preprocess in PREPROCESS_VARIANTS:
        candidate = _run_ocr(preprocess(crop), field, name)

        if best is None or candidate.weight > best.weight:
            best = candidate

        if candidate.parts and candidate.mean_confidence >= EARLY_EXIT_CONFIDENCE:
            break

    if best is None:
        return []

    logger.debug("%s chosen variant: %s", field, best.variant)
    return best.parts


# ---------------------------------------------------------------------------
# Trích xuất từng trường
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    """Tách từ, hạ chữ thường, bỏ dấu để so khớp không phụ thuộc OCR có đọc đúng dấu hay không."""
    decomposed = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return set(re.findall(r"\w+", stripped))


def extract_full_name(image: np.ndarray) -> str | None:
    parts = _ocr_region(image, "full_name", BOX_FULL_NAME)
    return " ".join(parts) or None


def extract_date_of_birth(image: np.ndarray) -> str | None:
    parts = _ocr_region(image, "date_of_birth", BOX_DATE_OF_BIRTH)
    if not parts:
        return None

    text = " ".join(parts)
    match = DATE_PATTERN.search(text)
    if match:
        return re.sub(r"\s+", "", match.group(0))
    return parts[0]


def extract_gender(image: np.ndarray) -> str | None:
    tokens = _tokens(" ".join(_ocr_region(image, "gender", BOX_GENDER)))

    if "nam" in tokens:
        return "Nam"
    if "nu" in tokens:
        return "Nữ"
    return None


def extract_nationality(image: np.ndarray) -> str | None:
    tokens = _tokens(" ".join(_ocr_region(image, "nationality", BOX_NATIONALITY)))

    # Chấp nhận cả khi OCR mất dấu, dính chữ ("VietNam") hoặc chỉ đọc được một trong hai từ
    if "nam" in tokens or any(token.startswith("viet") for token in tokens):
        return "Việt Nam"
    return None


def extract_place_of_origin(image: np.ndarray) -> str | None:
    parts = _ocr_region(image, "place_of_origin", BOX_PLACE_OF_ORIGIN)
    return " ".join(parts) or None


def extract_residence(image: np.ndarray) -> str | None:
    line_1 = " ".join(_ocr_region(image, "residence_line_1", BOX_RESIDENCE_LINE_1))
    line_2 = " ".join(_ocr_region(image, "residence_line_2", BOX_RESIDENCE_LINE_2))

    lines = [line for line in (line_1, line_2) if line]
    return ", ".join(lines) or None


def extract_id_number(image: np.ndarray) -> str | None:
    """Trích xuất số CCCD (12 chữ số) từ mặt trước thẻ.

    Xử lý các lỗi OCR phổ biến: nhầm O/0, I/1, l/1, S/5, B/8.
    Chỉ trả kết quả khi tìm được đúng 12 chữ số liên tiếp.
    Không tự sinh hoặc đoán chữ số còn thiếu.
    """
    parts = _ocr_region(image, "id_number", BOX_ID_NUMBER)
    if not parts:
        logger.debug("id_number: không có kết quả OCR")
        return None

    # Ghép toàn bộ text, áp dụng sửa lỗi ký tự, giữ lại chỉ chữ số
    raw = " ".join(parts)
    corrected = raw.translate(_ID_NUMBER_DIGIT_FIXES)
    digits_only = re.sub(r"\D", "", corrected)

    match = _ID_12_PATTERN.search(digits_only)
    if match:
        return match.group(0)

    logger.debug("id_number: không tìm được chuỗi 12 chữ số (raw=%r, digits=%r)", raw, digits_only)
    return None


def extract_expiry_date(image: np.ndarray) -> str | None:
    """Trích xuất trường "Có giá trị đến" từ ảnh CCCD.

    Hạn chế: trên CCCD chip mới (2021+), trường này nằm ở MẶT SAU.
    Nếu ảnh đầu vào là mặt trước, hàm sẽ không tìm được ngày và trả None.

    Chuẩn hóa kết quả về định dạng DD/MM/YYYY.
    Kiểm tra ngày tháng hợp lệ bằng datetime. Không tự tính từ ngày sinh.
    """
    parts = _ocr_region(image, "expiry_date", BOX_EXPIRY_DATE)
    if not parts:
        logger.debug("expiry_date: không có kết quả OCR (có thể ảnh là mặt trước)")
        return None

    text = " ".join(parts)
    match = DATE_PATTERN.search(text)
    if not match:
        logger.debug("expiry_date: không khớp pattern ngày (raw=%r)", text)
        return None

    # Chuẩn hoá: loại khoảng trắng thừa, thay phân cách về "/"
    raw_date = re.sub(r"\s+", "", match.group(0))
    normalized = re.sub(r"[.\-]", "/", raw_date)

    # Kiểm tra ngày hợp lệ
    try:
        datetime.strptime(normalized, "%d/%m/%Y")
    except ValueError:
        logger.debug("expiry_date: ngày không hợp lệ (%r)", normalized)
        return None

    return normalized


def extract_cccd_fields(image_path: str) -> dict[str, str | None]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = _normalize_width(image)

    return {
        "id_number": extract_id_number(image),
        "full_name": extract_full_name(image),
        "date_of_birth": extract_date_of_birth(image),
        "gender": extract_gender(image),
        "nationality": extract_nationality(image),
        "place_of_origin": extract_place_of_origin(image),
        "residence": extract_residence(image),
        "expiry_date": extract_expiry_date(image),
    }


if __name__ == "_main_":
    import json
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m services.cccd_extractor <image_path>")

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    print(json.dumps(extract_cccd_fields(sys.argv[1]), ensure_ascii=False, indent=2))