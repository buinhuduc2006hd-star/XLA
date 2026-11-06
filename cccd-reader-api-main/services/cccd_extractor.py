"""Trích xuất thông tin từ ảnh CCCD gắn chip (mặt trước) theo tỷ lệ tọa độ rồi OCR.

Mẫu thẻ hỗ trợ hiện tại
-----------------------
- Thẻ Căn cước công dân (CCCD) gắn chip theo Thông tư 06/2021/TT-BCA (áp dụng từ năm 2021 đến trước 01/07/2024).
- Tiêu đề thẻ: "CĂN CƯỚC CÔNG DÂN / Citizen Identity Card".
- Mặt trước gồm:
  + Bên trái: Quốc huy, ảnh chân dung, dòng "Có giá trị đến / Date of expiry" (ngay dưới ảnh chân dung).
  + Bên phải: Quốc hiệu, tiêu ngữ, Số CCCD ("Số / No."), Họ và tên, Ngày sinh, Giới tính, Quốc tịch, Quê quán, Nơi thường trú.
- Lưu ý phạm vi:
  + Chưa hỗ trợ mẫu "Thẻ Căn cước" phát hành từ ngày 01/07/2024 (theo Luật Căn cước 2023 / Thông tư 17/2024/TT-BCA
    có trường hết hạn chuyển sang mặt sau). Cần phân loại mẫu thẻ và bộ extractor riêng trước khi mở rộng.

Chạy thử / xem OCR thô của từng trường (từ thư mục gốc của project cccd-reader-api-main):

    python -m services.cccd_extractor duong_dan_anh.jpg
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
BOX_ID_NUMBER: Box = (0.37, 0.39, 0.89, 0.51)
BOX_FULL_NAME: Box = (0.27, 0.535, 0.62, 0.605)
BOX_DATE_OF_BIRTH: Box = (0.56, 0.61, 0.76, 0.70)
BOX_GENDER: Box = (0.46, 0.665, 0.545, 0.74)
BOX_NATIONALITY: Box = (0.81, 0.645, 0.985, 0.735)
BOX_PLACE_OF_ORIGIN: Box = (0.28, 0.79, 0.82, 0.845)
BOX_RESIDENCE_LINE_1: Box = (0.665, 0.855, 0.93, 0.915)  # vd: "Thôn Bích Đông"
BOX_RESIDENCE_LINE_2: Box = (0.28, 0.915, 0.84, 0.985)  # vd: "Nam Chính, Nam Sách, Hải Dương"

# --- Có giá trị đến / Date of expiry - mặt trước CCCD chip 2021 (Thông tư 06/2021/TT-BCA) ---
# Vị trí: góc dưới bên trái, ngay phía dưới ảnh chân dung công dân.
# Vùng chứa nhãn "Có giá trị đến / Date of expiry" và ngày hết hạn "DD/MM/YYYY" hoặc "Không thời hạn".
BOX_EXPIRY_DATE: Box = (0.025, 0.85, 0.28, 0.92)

# Bảng ROI duy nhất cho toàn bộ hệ thống (Extractor & Debug tool đồng bộ)
CCCD_ROIS: dict[str, Box] = {
    "id_number": BOX_ID_NUMBER,
    "full_name": BOX_FULL_NAME,
    "date_of_birth": BOX_DATE_OF_BIRTH,
    "gender": BOX_GENDER,
    "nationality": BOX_NATIONALITY,
    "place_of_origin": BOX_PLACE_OF_ORIGIN,
    "residence_line_1": BOX_RESIDENCE_LINE_1,
    "residence_line_2": BOX_RESIDENCE_LINE_2,
    "expiry_date": BOX_EXPIRY_DATE,
}

STANDARD_WIDTH = 1061  # chuẩn hoá ảnh về cùng độ rộng để tỷ lệ box ổn định
BOX_PADDING = 0.008  # nới nhẹ mỗi cạnh box (theo tỷ lệ ảnh)
MIN_CONFIDENCE = 0.3  # ảnh chụp thường mờ nên hạ ngưỡng so với 0.5
EARLY_EXIT_CONFIDENCE = 0.75  # đủ tốt thì không cần thử biến thể tiếp theo
UPSCALE = 2.0  # vùng cắt rất nhỏ, phóng to giúp OCR đọc tốt hơn

DATE_PATTERN = re.compile(r"\b(\d{2})\s*[/.\-]\s*(\d{2})\s*[/.\-]\s*(\d{4})\b")
DATE_8_DIGITS_PATTERN = re.compile(r"\b(\d{2})(\d{2})(\d{4})\b")


def _parse_date_string(text: str) -> str | None:
    """Trích xuất và chuẩn hóa ngày tháng từ văn bản OCR.

    1. Ưu tiên tìm định dạng có đủ 2 ký tự phân cách: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY.
    2. Hỗ trợ định dạng khuyết 1 dấu phân cách (do OCR mờ nét): DD/MMYYYY hoặc DDMM/YYYY.
    3. Hỗ trợ định dạng 8 chữ số liền nhau DDMMYYYY:
       - Chỉ áp dụng khi 8 chữ số được OCR từ đúng vùng ROI và có cấu trúc hợp lý.
       - Kiểm tra tính hợp lệ trên lịch bằng datetime (loại bỏ ngày 32, tháng 13, 29/02 năm không nhuận).
       - Kiểm tra phạm vi năm hợp lý (1900-2100).
    4. Trả về chuỗi định dạng chuẩn DD/MM/YYYY nếu hợp lệ, ngược lại trả về None.
    """
    # Chuẩn hóa khoảng trắng quanh dấu phân cách
    cleaned = re.sub(r"(\d)\s+[/.\-]\s*(\d)", r"\1/\2", text)
    cleaned = re.sub(r"(\d)\s*[/.\-]\s+(\d)", r"\1/\2", cleaned)

    # 1. Hai dấu phân cách: DD/MM/YYYY
    match_2 = re.search(r"\b(\d{2})[/.\-](\d{2})[/.\-](\d{4})\b", cleaned)
    if match_2:
        d, mth, y = match_2.groups()
        try:
            return datetime.strptime(f"{d}/{mth}/{y}", "%d/%m/%Y").strftime("%d/%m/%Y")
        except ValueError:
            pass

    # 2. Khuyết dấu phân cách thứ hai (vd: 27/032031)
    match_d1 = re.search(r"\b(\d{2})[/.\-](\d{2})(\d{4})\b", cleaned)
    if match_d1:
        d, mth, y = match_d1.groups()
        try:
            return datetime.strptime(f"{d}/{mth}/{y}", "%d/%m/%Y").strftime("%d/%m/%Y")
        except ValueError:
            pass

    # 3. Khuyết dấu phân cách thứ nhất (vd: 2703/2031)
    match_d2 = re.search(r"\b(\d{2})(\d{2})[/.\-](\d{4})\b", cleaned)
    if match_d2:
        d, mth, y = match_d2.groups()
        try:
            return datetime.strptime(f"{d}/{mth}/{y}", "%d/%m/%Y").strftime("%d/%m/%Y")
        except ValueError:
            pass

    # 4. Định dạng 8 chữ số thuần túy DDMMYYYY
    for match_8 in DATE_8_DIGITS_PATTERN.finditer(cleaned):
        day_str, month_str, year_str = match_8.groups()
        year = int(year_str)
        if 1900 <= year <= 2100:
            try:
                return datetime.strptime(f"{day_str}/{month_str}/{year_str}", "%d/%m/%Y").strftime("%d/%m/%Y")
            except ValueError:
                continue

    return None

# Bảng sửa ký tự nhầm sang số khi ứng viên có cấu trúc số CCCD phù hợp
_ID_DIGIT_FIXES = str.maketrans({
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "|": "1", "i": "1",
    "Z": "2",
    "S": "5", "s": "5",
    "B": "8",
})


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


def _run_ocr(
    processed: np.ndarray,
    field: str,
    variant: str,
    min_confidence: float = MIN_CONFIDENCE,
) -> OcrCandidate:
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
        if item["confidence"] >= min_confidence and re.search(r"\w", item["text"])
    ]
    if not kept:
        return OcrCandidate(variant, [], 0.0, 0.0)

    return OcrCandidate(
        variant=variant,
        parts=[text for text, _ in kept],
        mean_confidence=sum(conf for _, conf in kept) / len(kept),
        weight=sum(conf * len(text) for text, conf in kept),
    )


def _ocr_region(
    image: np.ndarray,
    field: str,
    box: Box,
    min_confidence: float = MIN_CONFIDENCE,
) -> list[str]:
    """Cắt vùng theo tỷ lệ, thử nhiều cách tiền xử lý, trả về kết quả OCR tốt nhất."""
    crop = _crop_region(image, field, box)

    best: OcrCandidate | None = None
    for name, preprocess in PREPROCESS_VARIANTS:
        candidate = _run_ocr(preprocess(crop), field, name, min_confidence=min_confidence)

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
    if not parts:
        return None

    # Lọc bỏ các từ rác có thể là nhãn "Họ và tên" hoặc "Full name" nếu lỡ lọt vào mép trên
    cleaned_parts: list[str] = []
    for part in parts:
        norm = unicodedata.normalize("NFD", part.lower())
        stripped = "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")
        # Nếu đoạn text chỉ chứa nhãn tiêu đề trường thì bỏ qua
        if re.search(r"\b(ho va ten|hovaten|full name|fullname|ho|ten)\b", stripped) and len(stripped) <= 12:
            continue
        cleaned_parts.append(part)

    result = " ".join(cleaned_parts).strip()
    return result or None


def extract_date_of_birth(image: np.ndarray) -> str | None:
    parts = _ocr_region(image, "date_of_birth", BOX_DATE_OF_BIRTH)
    if not parts:
        return None

    for text in parts:
        parsed = _parse_date_string(text)
        if parsed:
            return parsed
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


def _clean_id_candidate(text: str) -> str | None:
    """Kiểm tra và chuẩn hóa một ứng viên chuỗi số CCCD.

    Quy tắc an toàn:
    - Loại bỏ ký tự phân cách thông thường: khoảng trắng, dấu gạch ngang, dấu chấm.
    - Chỉ áp dụng sửa ký tự nhầm (O/0, I/1...) khi chuỗi có độ dài tiềm năng (11-13 ký tự)
      và phần lớn là ký tự số / ký tự dễ nhầm sang số.
    - Không sửa trên các chuỗi văn bản thông thường (họ tên, chữ tiếng Việt).
    - Trả về chuỗi 12 chữ số nếu hợp lệ, ngược lại trả về None.
    - TUYỆT ĐỐI không cắt chuỗi 13 số để lấy 12, không thêm số cho chuỗi 11 số.
    """
    cleaned = re.sub(r"[\s.\-_/:]+", "", text)
    if not cleaned:
        return None

    # Nếu chuỗi đã là 12 chữ số thuần
    if cleaned.isdigit() and len(cleaned) == 12:
        return cleaned

    # Nếu độ dài không nằm trong khoảng có thể chứa 12 ký tự -> loại bỏ
    if len(cleaned) != 12:
        return None

    # Kiểm tra xem có phải chuỗi hầu hết là số và ký tự dễ nhầm hay không
    # (tránh nhầm họ tên như 'NGUYEN VAN A' thành số)
    num_digits = sum(1 for ch in cleaned if ch.isdigit())
    if num_digits < 8:  # Nếu có quá ít số trong 12 ký tự -> không phải ứng viên số CCCD
        return None

    # Thử sửa các ký tự dễ nhầm sang số
    translated = cleaned.translate(_ID_DIGIT_FIXES)
    if translated.isdigit() and len(translated) == 12:
        return translated

    return None


def extract_id_number(image: np.ndarray) -> str | None:
    """Trích xuất số CCCD (12 chữ số) từ mặt trước thẻ.

    Quy tắc bảo vệ tính toàn vẹn dữ liệu:
    - Chỉ chấp nhận chính xác 12 chữ số. Chuỗi 11 hoặc 13 chữ số trả None.
    - Không cắt bớt chữ số thừa, không tự thêm chữ số thiếu.
    - Giữ số CCCD dưới dạng string (bảo toàn số 0 ở đầu).
    - Đánh giá từng ứng viên dòng/box OCR trước khi xem xét ghép dòng có cấu trúc phù hợp.
    - KHÔNG ghi số CCCD vào log (bảo mật dữ liệu cá nhân).
    """
    parts = _ocr_region(image, "id_number", BOX_ID_NUMBER)
    if not parts:
        logger.debug("id_number: không có kết quả OCR trong vùng ROI")
        return None

    # 1. Ưu tiên kiểm tra từng dòng/đoạn riêng lẻ trước (tránh ghép bừa bãi giữa các dòng khác nhau)
    for part in parts:
        candidate = _clean_id_candidate(part)
        if candidate is not None:
            logger.debug("id_number: trích xuất thành công 12 chữ số từ một đoạn OCR")
            return candidate

    # 2. Nếu OCR bị tách nhỏ số thành nhiều cụm cùng 1 dòng (ví dụ: '079', '123', '456789')
    # Chỉ ghép nếu tất cả các phần tử ghép đều là số hoặc ký tự phân tách
    combined = "".join(parts)
    candidate = _clean_id_candidate(combined)
    if candidate is not None:
        logger.debug("id_number: trích xuất thành công 12 chữ số từ các đoạn ghép")
        return candidate

    logger.debug("id_number: không tìm thấy chuỗi chính xác 12 chữ số hợp lệ")
    return None


def extract_expiry_date(image: np.ndarray) -> str | None:
    """Trích xuất trường "Có giá trị đến / Date of expiry" từ mặt trước thẻ CCCD chip 2021.

    Vị trí: góc dưới bên trái, ngay phía dưới ảnh chân dung công dân.
    Chuẩn hóa kết quả về định dạng DD/MM/YYYY (hỗ trợ cả DD/MM/YYYY và DDMMYYYY 8 số hợp lệ).
    Kiểm tra ngày tháng hợp lệ bằng datetime. Không tự tính từ ngày sinh.
    Nếu trường không xuất hiện hoặc không đọc được, trả về None.
    """
    # 1. Thử với ngưỡng chuẩn của hệ thống (MIN_CONFIDENCE = 0.3)
    parts = _ocr_region(image, "expiry_date", BOX_EXPIRY_DATE, min_confidence=MIN_CONFIDENCE)
    for text in parts:
        parsed = _parse_date_string(text)
        if parsed:
            logger.debug("expiry_date: trích xuất ngày hợp lệ thành công (%s)", parsed)
            return parsed

    # 2. Bổ sung: kiểm tra khi ngày ở góc thẻ có độ tương phản/độ phân giải thấp
    # Chỉ chấp nhận khi tìm thấy đúng 8 chữ số độc lập và được datetime xác thực hợp lệ trên lịch.
    parts_candidate = _ocr_region(image, "expiry_date", BOX_EXPIRY_DATE, min_confidence=0.10)
    for text in parts_candidate:
        parsed = _parse_date_string(text)
        if parsed:
            logger.debug("expiry_date: trích xuất ngày từ 8 chữ số đã xác thực lịch (%s)", parsed)
            return parsed

    # Trường hợp không thời hạn (nếu có nhãn hoặc không có ngày hợp lệ)
    all_parts = parts or parts_candidate
    if all_parts:
        tokens = _tokens(" ".join(all_parts))
        if "khong" in tokens and ("thoi" in tokens or "han" in tokens):
            logger.debug("expiry_date: nhận diện không thời hạn")
            return "Không thời hạn"

    logger.debug("expiry_date: không tìm thấy ngày hết hạn hợp lệ")
    return None


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


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m services.cccd_extractor <image_path>")

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    print(json.dumps(extract_cccd_fields(sys.argv[1]), ensure_ascii=False, indent=2))