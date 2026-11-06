"""Tự nhận diện loại giấy tờ (CCCD / GPLX) rồi gọi extractor phù hợp.

Chạy thử (từ thư mục gốc của project):

    python -m services.id_card_extractor duong_dan_anh.jpg
"""

from __future__ import annotations

import logging
from typing import Literal

import cv2

from services.cccd_extractor import (
    Box,
    _normalize_width,
    _ocr_region,
    _tokens,
    extract_cccd_fields,
)
from services.gplx_extractor import extract_gplx_fields

logger = logging.getLogger(__name__)

CardType = Literal["cccd", "gplx"]

# Phần trên của thẻ: chứa tiêu đề "CĂN CƯỚC CÔNG DÂN" hoặc "GIẤY PHÉP LÁI XE"
BOX_TITLE: Box = (0.0, 0.0, 1.0, 0.42)

# Từ khoá đã bỏ dấu, chỉ dùng từ đặc trưng của từng loại.
# Không dùng "cong" / "dan": "CỘNG HÒA" ở đầu cả hai loại thẻ cũng chứa các từ gần giống.
CCCD_KEYWORDS: frozenset[str] = frozenset({"cuoc", "citizen", "identity"})
GPLX_KEYWORDS: frozenset[str] = frozenset({"giay", "phep", "driver", "license"})


def _score(tokens: set[str], keywords: frozenset[str]) -> int:
    # Dùng "in" để chấp nhận OCR dính chữ (vd: "GIAYPHEPLAIXE", "DRIVERS")
    return sum(1 for keyword in keywords if any(keyword in token for token in tokens))


def detect_card_type(image_path: str) -> CardType | None:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = _normalize_width(image)
    tokens = _tokens(" ".join(_ocr_region(image, "card_title", BOX_TITLE)))

    cccd_score = _score(tokens, CCCD_KEYWORDS)
    gplx_score = _score(tokens, GPLX_KEYWORDS)
    logger.debug("card type scores: cccd=%d gplx=%d tokens=%s", cccd_score, gplx_score, sorted(tokens))

    if cccd_score == gplx_score:
        return None
    return "cccd" if cccd_score > gplx_score else "gplx"


def extract_id_card_fields(image_path: str) -> dict[str, str | None]:
    """Trả về dict gồm card_type ("cccd" | "gplx") và các trường tương ứng với loại thẻ."""
    card_type = detect_card_type(image_path)

    if card_type == "cccd":
        return {"card_type": "cccd", **extract_cccd_fields(image_path)}
    if card_type == "gplx":
        return {"card_type": "gplx", **extract_gplx_fields(image_path)}

    raise ValueError("Không nhận diện được loại giấy tờ (cần CCCD hoặc GPLX)")


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m services.id_card_extractor <image_path>")

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    print(json.dumps(extract_id_card_fields(sys.argv[1]), ensure_ascii=False, indent=2))