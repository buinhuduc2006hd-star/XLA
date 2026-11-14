"""Adaptive OCR Extractor với Field-Level Fallback & Metrics (Giai đoạn 3.2 - Pipeline E).

Các thành phần chính:
1. Field Validators: Kiểm tra tính hợp lệ về định dạng/nội dung của từng trường.
2. Field-Level Fallback: Tự động fallback có kiểm soát (tối đa 1 lần retry/trường)
   khi trường trích xuất ban đầu bị thiếu hoặc không đạt chuẩn validation.
3. Metric Evaluators: Tính toán Exact Match, Field Extraction Rate,
   Field Validation Rate, CER (Character Error Rate), WER (Word Error Rate),
   và Document-Level Accuracy.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from services.calibrated_rois import (
    RECTIFIED_CCCD_ROIS,
    RECTIFIED_GPLX_ROIS,
    extract_calibrated_cccd_fields,
    extract_calibrated_gplx_fields,
    _crop_region,
    _normalize_width,
    _preprocess_dot_matrix,
    _preprocess_green_channel,
    _ocr_region_smart,
)
from services.cccd_extractor import (
    _clean_id_candidate,
    _parse_date_string,
    _tokens,
    DATE_PATTERN,
    extract_id_number as extract_cccd_id_baseline,
    extract_full_name as extract_cccd_name_baseline,
    extract_date_of_birth as extract_cccd_dob_baseline,
    extract_gender as extract_cccd_gender_baseline,
    extract_nationality as extract_cccd_nat_baseline,
    extract_place_of_origin as extract_cccd_origin_baseline,
    extract_residence as extract_cccd_res_baseline,
    extract_expiry_date as extract_cccd_exp_baseline,
)
from services.gplx_extractor import (
    _NUMBER_DIGIT_FIXES,
    VALID_CLASSES,
    _normalize_class,
    extract_license_number as extract_gplx_lic_baseline,
    extract_full_name as extract_gplx_name_baseline,
    extract_date_of_birth as extract_gplx_dob_baseline,
    extract_nationality as extract_gplx_nat_baseline,
    extract_address as extract_gplx_addr_baseline,
    extract_license_class as extract_gplx_class_baseline,
    extract_expiry as extract_gplx_exp_baseline,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. FIELD VALIDATORS (KIỂM TRA CHẤT LƯỢNG ĐẦU RA CỦA TRƯỜNG)
# ===========================================================================

def validate_id_number(val: str | None) -> bool:
    if not val:
        return False
    clean = re.sub(r"\D", "", val)
    return len(clean) == 12


def validate_license_number(val: str | None) -> bool:
    if not val:
        return False
    clean = re.sub(r"\D", "", val)
    return 9 <= len(clean) <= 12


def validate_full_name(val: str | None) -> bool:
    if not val:
        return False
    val_clean = val.strip()
    if len(val_clean) < 3 or any(ch.isdigit() for ch in val_clean):
        return False
    # Phải có ít nhất 2 từ
    words = val_clean.split()
    return len(words) >= 2


def validate_date(val: str | None) -> bool:
    if not val:
        return False
    return _parse_date_string(val) is not None


def validate_gender(val: str | None) -> bool:
    if not val:
        return False
    return val.strip().lower() in {"nam", "nữ", "nu"}


def validate_nationality(val: str | None) -> bool:
    if not val:
        return False
    val_lower = val.lower()
    toks = _tokens(val)
    return (
        "viet" in toks
        or "nam" in toks
        or any(t.startswith("viet") for t in toks)
        or "vietnam" in val_lower
        or "việt" in val_lower
        or "nam" in val_lower
    )


def validate_license_class(val: str | None) -> bool:
    if not val:
        return False
    return _normalize_class(val) in VALID_CLASSES


def validate_expiry(val: str | None) -> bool:
    if not val:
        return False
    val_lower = val.lower()
    if "không thời hạn" in val_lower or "khong thoi han" in val_lower or "khong" in val_lower:
        return True
    return validate_date(val)


CCCD_VALIDATORS: dict[str, Callable[[str | None], bool]] = {
    "id_number": validate_id_number,
    "full_name": validate_full_name,
    "date_of_birth": validate_date,
    "gender": validate_gender,
    "nationality": validate_nationality,
    "place_of_origin": lambda v: bool(v and len(v.strip()) >= 5),
    "residence": lambda v: bool(v and len(v.strip()) >= 5),
    "expiry_date": validate_expiry,
}

GPLX_VALIDATORS: dict[str, Callable[[str | None], bool]] = {
    "license_number": validate_license_number,
    "full_name": validate_full_name,
    "date_of_birth": validate_date,
    "nationality": validate_nationality,
    "address": lambda v: bool(v and len(v.strip()) >= 5),
    "license_class": validate_license_class,
    "expiry": validate_expiry,
}


# ===========================================================================
# 2. CER & WER EVALUATION METRICS (LEVENSHTEIN DISTANCE)
# ===========================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Khoảng cách Levenshtein giữa 2 chuỗi (ký tự hoặc từ)."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_cer(prediction: str | None, reference: str | None) -> float:
    """Character Error Rate: EditDistance(pred, ref) / max(len(ref), 1)."""
    if reference is None:
        return 0.0 if prediction is None else 1.0
    ref_norm = " ".join(reference.strip().split())
    if prediction is None:
        return 1.0
    pred_norm = " ".join(prediction.strip().split())
    if not ref_norm:
        return 0.0 if not pred_norm else 1.0

    dist = levenshtein_distance(pred_norm, ref_norm)
    return round(dist / len(ref_norm), 4)


def compute_wer(prediction: str | None, reference: str | None) -> float:
    """Word Error Rate: WordEditDistance(pred_words, ref_words) / max(len(ref_words), 1)."""
    if reference is None:
        return 0.0 if prediction is None else 1.0
    ref_words = reference.strip().split()
    if prediction is None:
        return 1.0
    pred_words = prediction.strip().split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0

    # Dùng thuật toán Levenshtein trên danh sách từ
    m, n = len(pred_words), len(ref_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred_words[i - 1].lower() == ref_words[j - 1].lower() else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return round(dp[m][n] / len(ref_words), 4)


def is_exact_match(prediction: str | None, reference: str | None) -> bool:
    """So khớp chính xác sau khi chuẩn hóa khoảng trắng và hạ chữ thường nếu cần."""
    if prediction is None or reference is None:
        return prediction == reference
    p_norm = " ".join(prediction.strip().upper().split())
    r_norm = " ".join(reference.strip().upper().split())
    return p_norm == r_norm


# ===========================================================================
# 3. PIPELINE E: ADAPTIVE OCR EXTRACTOR VỚI FIELD-LEVEL FALLBACK
# ===========================================================================

@dataclass
class ExtractionResult:
    fields: dict[str, str | None]
    valid_flags: dict[str, bool]
    fallback_used: dict[str, bool]
    fallback_sources: dict[str, str | None]
    total_fallbacks: int = 0
    successful_fallbacks: int = 0


def _extract_single_baseline_field(
    yolo_crop: np.ndarray,
    card_type: str,
    field_name: str,
) -> str | None:
    """Trích xuất một trường đơn lẻ từ YOLO crop bằng baseline extractor."""
    norm = _normalize_width(yolo_crop)
    if card_type == "cccd":
        extractors = {
            "id_number": extract_cccd_id_baseline,
            "full_name": extract_cccd_name_baseline,
            "date_of_birth": extract_cccd_dob_baseline,
            "gender": extract_cccd_gender_baseline,
            "nationality": extract_cccd_nat_baseline,
            "place_of_origin": extract_cccd_origin_baseline,
            "residence": extract_cccd_res_baseline,
            "expiry_date": extract_cccd_exp_baseline,
        }
    else:
        extractors = {
            "license_number": extract_gplx_lic_baseline,
            "full_name": extract_gplx_name_baseline,
            "date_of_birth": extract_gplx_dob_baseline,
            "nationality": extract_gplx_nat_baseline,
            "address": extract_gplx_addr_baseline,
            "license_class": extract_gplx_class_baseline,
            "expiry": extract_gplx_exp_baseline,
        }
    func = extractors.get(field_name)
    if func:
        try:
            return func(norm)
        except Exception as e:
            logger.debug(f"Baseline extraction for {field_name} failed: {type(e).__name__}")
            return None
    return None


def extract_adaptive_pipeline_e(
    rectified_image: np.ndarray,
    yolo_crop: np.ndarray,
    card_type: str,
    mode: str = "accurate",
    max_fallbacks_per_doc: int = 3,
    max_fallback_time_sec: float = 60.0,
) -> ExtractionResult:
    """Pipeline E: Calibrated ROIs + Smart Early-Exit + Field-Level Fallback.

    Args:
        rectified_image: Ảnh đã nắn phẳng (856x540 hoặc crop).
        yolo_crop: Ảnh crop gốc từ YOLO bounding box.
        card_type: 'cccd' hoặc 'gplx'.
        mode: 'fast' (1 pass, không retry/fallback) hoặc 'accurate' (adaptive prep, bounded retry).
        max_fallbacks_per_doc: Ngân sách số lần fallback tối đa cho một tài liệu (ngăn ngừa runaway latency).
        max_fallback_time_sec: Thời gian tối đa dành cho fallback trước khi ngắt an toàn.
    """
    t_start = time.perf_counter()
    is_fast = (str(mode).lower() == "fast")

    # Bước 1: Trích xuất chính trên ảnh Rectified bằng Calibrated ROIs
    if card_type == "cccd":
        initial_fields = extract_calibrated_cccd_fields(rectified_image, fast_mode=is_fast)
        validators = CCCD_VALIDATORS
    else:
        initial_fields = extract_calibrated_gplx_fields(rectified_image, fast_mode=is_fast)
        validators = GPLX_VALIDATORS

    final_fields = dict(initial_fields)
    valid_flags: dict[str, bool] = {}
    fallback_used: dict[str, bool] = {}
    fallback_sources: dict[str, str | None] = {}

    total_fallbacks = 0
    successful_fallbacks = 0

    # Bước 2: Duyệt từng trường để kiểm tra validation và kích hoạt Fallback nếu cần
    for fld, val in initial_fields.items():
        validator = validators.get(fld, lambda v: v is not None)
        is_valid = validator(val)

        if is_valid:
            valid_flags[fld] = True
            fallback_used[fld] = False
            fallback_sources[fld] = "rectified"
        else:
            if is_fast:
                # Fast mode: không thử lại, ưu tiên latency tối đa
                valid_flags[fld] = False
                fallback_used[fld] = False
                fallback_sources[fld] = "rectified" if val else "unextracted"
            else:
                # Accurate mode: Kiểm tra ngân sách xử lý (Bounded Retry & Latency Guard)
                elapsed_sec = time.perf_counter() - t_start
                if total_fallbacks >= max_fallbacks_per_doc or elapsed_sec >= max_fallback_time_sec:
                    logger.warning(
                        f"Fallback budget exceeded (used={total_fallbacks}/{max_fallbacks_per_doc}, "
                        f"elapsed={elapsed_sec:.1f}s/{max_fallback_time_sec}s). "
                        f"Skipping fallback for field '{fld}' to prevent latency runaway."
                    )
                    valid_flags[fld] = False
                    fallback_used[fld] = False
                    fallback_sources[fld] = "rectified" if val else "unextracted"
                    continue

                total_fallbacks += 1
                fallback_used[fld] = True
                logger.debug(f"Triggering field fallback for '{fld}' (initial_valid=False, has_value={bool(val)})")

                # Fallback: Trích xuất từ YOLO Crop qua baseline extractor
                recovered_val = _extract_single_baseline_field(yolo_crop, card_type, fld)

                if validator(recovered_val):
                    final_fields[fld] = recovered_val
                    valid_flags[fld] = True
                    fallback_sources[fld] = "yolo_crop_baseline"
                    successful_fallbacks += 1
                    logger.debug(f"Field fallback SUCCEEDED for '{fld}'")
                else:
                    # Nếu vẫn không đạt validator, giữ giá trị tốt hơn giữa val và recovered_val
                    final_fields[fld] = val or recovered_val
                    valid_flags[fld] = False
                    fallback_sources[fld] = "failed"

    return ExtractionResult(
        fields=final_fields,
        valid_flags=valid_flags,
        fallback_used=fallback_used,
        fallback_sources=fallback_sources,
        total_fallbacks=total_fallbacks,
        successful_fallbacks=successful_fallbacks,
    )
