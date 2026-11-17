"""Bộ công cụ làm sạch dữ liệu OCR nhận thức theo trường (Field-Aware Postprocessing).

Giai đoạn P1: Xử lý triệt để hiện tượng lẫn nhãn song ngữ (Việt/Anh) và ký tự rác OCR
cho thẻ Căn cước công dân (CCCD) và Giấy phép lái xe (GPLX).

Nguyên tắc thiết kế:
1. Phân biệt rõ giữa Dòng nhãn (Header label) và Dòng giá trị (Value line).
2. Xóa bỏ an toàn các biến thể OCR sai lệch (vd: LEN / NAMNE, Place oÍ origin, Placo of residence).
3. Tuyệt đối không xóa nhầm các từ hợp lệ trong tên người (Hồ, Nam, Quê, Quán...) hoặc địa chỉ.
4. Giữ nguyên chuỗi OCR thô (raw_ocr) không thay đổi để phục vụ kiểm toán và hiển thị.
5. Cung cấp hàm kiểm tra nhiễm bẩn nhãn (has_label_contamination) cho Validator.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. BIỂU THỨC CHÍNH QUY CHO CÁC NHÃN TIÊU ĐỀ VÀ NHIỄU OCR
# ===========================================================================

# Tiêu đề trường Họ và tên
# Bao gồm: Họ và tên, Họ tên, Full name, Ho va ten, LEN / NAMNE, LEN/NAME, Fud name...
_RE_NAME_LABELS = re.compile(
    r"^\s*(?:"
    r"(?:Họ\s*(?:và|v\S*)?\s*tên|Họ\s*tên|Ho\s*va\s*ten|LEN\b|FULL\s*NAME?|NAMNE|HO\s*VA\s*TEN)"
    r"\s*(?:[/:\-\\_|]\s*(?:Full\s*name?|NAMNE?|NAME|Fud\s*name?|Họ\s*tên|Tên|TEN)?)*"
    r"\s*[/:\-\\_|]*\s*"
    r")\s*",
    re.IGNORECASE,
)

# Nhận diện dòng chỉ thuần túy là nhãn họ và tên
_RE_PURE_NAME_LABEL = re.compile(
    r"^\s*(?:Họ\s*(?:và|v\S*)?\s*tên|Họ\s*tên|Ho\s*va\s*ten|Full\s*name|LEN\s*[/|\\]*\s*NAMNE?|HO\s*VA\s*TEN|NAME|NAMNE)\s*[/:\-\\_|]*\s*$",
    re.IGNORECASE,
)

# Tiêu đề trường Quê quán
# Bao gồm: Quê quán, Quê quán / Place of origin, Place oÍ origin, Que 1, Place of origin...
_RE_ORIGIN_LABELS = re.compile(
    r"^\s*(?:"
    r"(?:Quê\s*quán|Que\s*quan|Que\s*1|Que\b|Place\s*o[fíiÍìï]\s*origi?n?|Place\s*of\s*origi?n?|origi?n?\b)"
    r"\s*[/:\-\\_|]*\s*"
    r")+\s*",
    re.IGNORECASE,
)

_RE_PURE_ORIGIN_LABEL = re.compile(
    r"^\s*(?:Quê\s*quán|Que\s*quan|Place\s*o[fíiÍìï]\s*origi?n?|Place\s*of\s*origi?n?|Que\s*1\s*Place\s*o[fíiÍìï]\s*origin)\s*[/:\-\\_|]*\s*$",
    re.IGNORECASE,
)

# Từ 'quan' hoặc 'quán' mồ côi nằm ở cuối chuỗi quê quán (do nhãn Quê quán bị OCR ngắt sang cuối)
_RE_TRAILING_QUAN = re.compile(r"\s*[,/|\-\\_]*\s*qu[aá]n\s*$", re.IGNORECASE)

# Tiền tố nhầm từ dòng Quốc tịch phía trên lọt vào Quê quán
_RE_ORIGIN_BLEED_TOP = re.compile(
    r"^\s*(?:Quốc\s*tịch|Nationality|Việt\s*Nam)\s*[/:\-\\_|]*\s*",
    re.IGNORECASE,
)

# Tiêu đề trường Nơi thường trú
# Bao gồm: Nơi thường trú, Place of residence, Placo of residence, J trú / Placo of residence, 1 trú...
_RE_RESIDENCE_LABELS = re.compile(
    r"^\s*(?:"
    r"(?:Nơi\s*thường\s*trú|Noi\s*thuong\s*tru|thường\s*trú|thuong\s*tru|[J1iI]\s*trú\b|Place\s*of\s*residence|Placo\s*of\s*residence|Place\s*of\s*rosidence|tesidercu|residence\b|rosidence\b)"
    r"\s*[/:\-\\_|]*\s*"
    r")+\s*",
    re.IGNORECASE,
)

_RE_PURE_RESIDENCE_LABEL = re.compile(
    r"^\s*(?:Nơi\s*thường\s*trú|Noi\s*thuong\s*tru|Place\s*of\s*residence|Placo\s*of\s*residence|[J1iI]\s*trú\s*[/|\\]*\s*Placo\s*of\s*residence)\s*[/:\-\\_|]*\s*$",
    re.IGNORECASE,
)

# Nhiễu ký tự OCR đơn lẻ giữa các phân đoạn địa chỉ (vd: ', L Nam Chinh' -> ', Nam Chinh')
_RE_STRAY_MID_NOISE = re.compile(r",\s*[LIJ1]\s+", re.IGNORECASE)


# ===========================================================================
# 2. HÀM LÀM SẠCH TỪNG TRƯỜNG DỮ LIỆU
# ===========================================================================

def clean_full_name(
    raw_text: str | None,
    lines: list[Any] | None = None,
) -> tuple[str | None, str | None, bool]:
    """Làm sạch trường Họ và tên.

    Loại bỏ nhãn 'Họ và tên / Full name' và các biến thể OCR lỗi ('LEN / NAMNE', 'NAME').
    Tuyệt đối bảo vệ tên thật có chứa các từ như 'HỒ', 'NAM', 'QUÂN'.

    Args:
        raw_text: Chuỗi thô nhận được từ OCR.
        lines: Danh sách các dòng OCR (nếu có thông tin tọa độ / bounding box).

    Returns:
        tuple (cleaned_name, raw_name, is_clean):
        - cleaned_name: Tên chuẩn hóa in hoa hoặc None nếu rỗng.
        - raw_name: Văn bản thô ban đầu.
        - is_clean: True nếu dữ liệu hợp lệ và không còn dấu vết nhiễm bẩn nhãn.
    """
    if not raw_text:
        return None, None, True

    raw_clean = raw_text.strip()
    if not raw_clean:
        return None, None, True

    # 1. Nếu có thông tin dòng OCR, loại bỏ các dòng thuần nhãn
    candidate_text = raw_clean
    if lines:
        kept_lines = []
        for line in lines:
            line_str = getattr(line, "text", str(line)).strip()
            if not line_str:
                continue
            # Nếu dòng chỉ chứa nhãn tiêu đề thì loại bỏ
            if _RE_PURE_NAME_LABEL.match(line_str):
                continue
            # Nếu dòng chứa cả nhãn ở đầu và tên ở sau, làm sạch tiền tố
            cleaned_line = _RE_NAME_LABELS.sub("", line_str).strip()
            if cleaned_line:
                kept_lines.append(cleaned_line)
        if kept_lines:
            candidate_text = " ".join(kept_lines)

    # 2. Xóa các biến thể nhãn tiền tố còn sót lại
    cleaned = _RE_NAME_LABELS.sub("", candidate_text).strip()

    # 3. Loại bỏ ký tự rác không thuộc chữ cái tiếng Việt (dấu gạch chéo, số, gạch dưới...)
    # Tên tiếng Việt chỉ chứa chữ cái và khoảng trắng
    cleaned = re.sub(r"[/\\|_\-~^:;*+?!<=>@#$%&()0-9]+", " ", cleaned)

    # 4. Chuẩn hóa khoảng trắng và chuyển chữ in hoa
    words = cleaned.split()
    cleaned = " ".join(words).upper()

    if not cleaned or len(cleaned) < 2:
        return None, raw_clean, False

    # 5. Đánh giá tính 'clean': không còn chứa từ khóa nhãn cấm
    has_contam = has_label_contamination("full_name", cleaned)
    is_clean = not has_contam and len(words) >= 2

    return cleaned, raw_clean, is_clean


def clean_place_of_origin(
    raw_text: str | None,
    lines: list[Any] | None = None,
) -> tuple[str | None, str | None, bool]:
    """Làm sạch trường Quê quán.

    Loại bỏ nhãn 'Quê quán / Place of origin', 'Que 1 Place oÍ origin', và các từ
    mồ côi ở đuôi chuỗi. Bảo vệ tuyệt đối địa danh có chứa chữ 'Quê' (vd: Xã Quê Mỹ).

    Returns:
        tuple (cleaned_origin, raw_origin, is_clean)
    """
    if not raw_text:
        return None, None, True

    raw_clean = raw_text.strip()
    if not raw_clean:
        return None, None, True

    candidate_text = raw_clean
    if lines:
        kept_lines = []
        for line in lines:
            line_str = getattr(line, "text", str(line)).strip()
            if not line_str:
                continue
            if _RE_PURE_ORIGIN_LABEL.match(line_str):
                continue
            cleaned_line = _RE_ORIGIN_LABELS.sub("", line_str).strip()
            if cleaned_line:
                kept_lines.append(cleaned_line)
        if kept_lines:
            candidate_text = " ".join(kept_lines)

    # 1. Xóa tiền tố nhãn quê quán
    cleaned = _RE_ORIGIN_LABELS.sub("", candidate_text).strip()

    # 2. Xóa tiền tố trôi từ trường Quốc tịch phía trên
    cleaned = _RE_ORIGIN_BLEED_TOP.sub("", cleaned).strip()

    # 3. Xóa từ 'quan' / 'quán' mồ côi nằm ở cuối chuỗi
    # Chỉ xóa nếu không có đơn vị hành chính đứng liền trước (vd: không phải 'Quận Ba Đình')
    if not re.search(r"\b(quận|huyện|thị xã|xã|phường)\s+qu[aá]n$", cleaned, re.IGNORECASE):
        cleaned = _RE_TRAILING_QUAN.sub("", cleaned).strip()

    # 4. Làm sạch dấu phân cách đầu/cuối và chuẩn hóa dấu phẩy
    cleaned = re.sub(r"^[,\-.:;/\\|\s]+", "", cleaned)
    cleaned = re.sub(r"[,\-.:;/\\|\s]+$", "", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned or len(cleaned) < 3:
        return None, raw_clean, False

    has_contam = has_label_contamination("place_of_origin", cleaned)
    is_clean = not has_contam and len(cleaned) >= 5

    return cleaned, raw_clean, is_clean


def clean_residence(
    raw_text: str | None,
    lines: list[Any] | None = None,
) -> tuple[str | None, str | None, bool]:
    """Làm sạch trường Nơi thường trú.

    Loại bỏ nhãn 'Nơi thường trú / Place of residence', 'J trú / Placo of residence',
    và các ký tự nhiễu giữa các dòng. Bảo tồn đầy đủ số nhà và địa danh.

    Returns:
        tuple (cleaned_residence, raw_residence, is_clean)
    """
    if not raw_text:
        return None, None, True

    raw_clean = raw_text.strip()
    if not raw_clean:
        return None, None, True

    candidate_text = raw_clean
    if lines:
        kept_lines = []
        for line in lines:
            line_str = getattr(line, "text", str(line)).strip()
            if not line_str:
                continue
            if _RE_PURE_RESIDENCE_LABEL.match(line_str):
                continue
            cleaned_line = _RE_RESIDENCE_LABELS.sub("", line_str).strip()
            if cleaned_line:
                kept_lines.append(cleaned_line)
        if kept_lines:
            candidate_text = ", ".join(kept_lines)

    # 1. Xóa tiền tố nhãn nơi thường trú
    cleaned = _RE_RESIDENCE_LABELS.sub("", candidate_text).strip()

    # 2. Xóa nhiễu ký tự đơn lẻ giữa các dòng (vd: ', L Nam Chinh')
    cleaned = _RE_STRAY_MID_NOISE.sub(", ", cleaned)

    # 3. Chuẩn hóa dấu phẩy và khoảng trắng
    cleaned = re.sub(r"^[,\-.:;/\\|\s]+", "", cleaned)
    cleaned = re.sub(r"[,\-.:;/\\|\s]+$", "", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned or len(cleaned) < 3:
        return None, raw_clean, False

    has_contam = has_label_contamination("residence", cleaned)
    is_clean = not has_contam and len(cleaned) >= 5

    return cleaned, raw_clean, is_clean


# ===========================================================================
# 3. KIỂM TRA NHIỄM BẨN NHÃN (LABEL CONTAMINATION DETECTOR)
# ===========================================================================

# Tập hợp các từ khóa nhãn cấm xuất hiện trong giá trị trường đã xử lý
FORBIDDEN_NAME_KEYWORDS = [
    "HỌ VÀ TÊN", "HO VA TEN", "FULL NAME", "FULLNAME", "NAMNE",
    "LEN /", "LEN/", "/ NAMNE", "FUD NAME", "HỌ TÊN", "HO TEN",
]

FORBIDDEN_ORIGIN_KEYWORDS = [
    "PLACE OF ORIGIN", "PLACE OÍ ORIGIN", "PLACE OÏ ORIGIN", "PLACE OF",
    "QUÊ QUÁN", "QUE QUAN", "PLACE OÍ", "PLACE OI",
]

FORBIDDEN_RESIDENCE_KEYWORDS = [
    "PLACE OF RESIDENCE", "PLACO OF RESIDENCE", "PLACE OF ROSIDENCE",
    "NƠI THƯỜNG TRÚ", "NOI THUONG TRU", "J TRÚ", "THƯỜNG TRÚ", "TESIDERCU",
]


def has_label_contamination(field_name: str, text: str | None) -> bool:
    """Kiểm tra xem chuỗi giá trị có chứa nhãn song ngữ hoặc ký tự lỗi nhãn hay không.

    Dùng cho Validator để từ chối valid=true nếu dữ liệu vẫn còn dính nhãn.

    Args:
        field_name: Tên trường ('full_name', 'place_of_origin', 'residence', ...)
        text: Giá trị cần kiểm tra.

    Returns:
        True nếu phát hiện có nhiễm bẩn nhãn, False nếu sạch sẽ.
    """
    if not text:
        return False

    norm_upper = unicodedata.normalize("NFD", text).upper()
    # Loại dấu để so khớp không phụ thuộc font/encoding
    norm_no_accents = "".join(ch for ch in norm_upper if unicodedata.category(ch) != "Mn")
    text_upper = text.upper()

    if field_name == "full_name":
        # 1. Kiểm tra các từ khóa nhãn cấm
        for kw in FORBIDDEN_NAME_KEYWORDS:
            if kw in text_upper or kw in norm_no_accents:
                return True
        # 2. Dấu phân cách nhãn đặc trưng
        if "/" in text or "\\" in text or "|" in text:
            return True
        # 3. Chứa chữ số
        if any(ch.isdigit() for ch in text):
            return True
        return False

    elif field_name in ("place_of_origin", "address"):
        for kw in FORBIDDEN_ORIGIN_KEYWORDS:
            if kw in text_upper or kw in norm_no_accents:
                return True
        # Không được chứa 'Place' hoặc 'origin' nếu đứng đầu
        if re.search(r"^\s*(?:Place|origin|Que\s*1)\b", text, re.IGNORECASE):
            return True
        return False

    elif field_name == "residence":
        for kw in FORBIDDEN_RESIDENCE_KEYWORDS:
            if kw in text_upper or kw in norm_no_accents:
                return True
        if re.search(r"^\s*(?:Place|residence|Placo|[J1iI]\s*trú)\b", text, re.IGNORECASE):
            return True
        return False

    elif field_name == "id_number":
        # Số CCCD chỉ chứa đúng 12 chữ số
        clean_digits = re.sub(r"\D", "", text)
        return len(clean_digits) != 12

    elif field_name in ("date_of_birth", "expiry_date", "expiry"):
        # Không được chứa các từ nhãn ngày sinh hay hết hạn
        if any(w in text_upper for w in ["NGÀY SINH", "DATE OF BIRTH", "CÓ GIÁ TRỊ ĐẾN", "EXPIRY", "EXP"]):
            return True
        return False

    return False
