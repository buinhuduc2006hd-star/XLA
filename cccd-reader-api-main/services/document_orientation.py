"""Module xác định và hiệu chỉnh chiều đọc văn bản (Orientation Correction) cho CCCD và GPLX.

Giai đoạn 3.1: Document Orientation Correction bằng OpenCV & Lightweight Anchor OCR.
- Hỗ trợ các góc xoay: 0°, 90°, 180°, 270°.
- Phân biệt rõ:
  + Geometric Rectification: nắn phẳng hình học về khung chữ nhật 856x540.
  + Orientation Correction: xác định chiều đọc văn bản (Text Upright Direction).
- Tối ưu CPU:
  + Không chạy full EasyOCR 4 lần trên toàn bộ ảnh.
  + Ưu tiên kiểm tra dải tiêu đề quốc hiệu (Anchor Header Strip: y in [0.0, 0.28]) giữa 0° và 180°.
  + Hỗ trợ kiểm tra 90° và 270° khi cần thiết (ảnh đứng / tỷ lệ bất thường).
  + Trả về OrientationResult với điểm số tin cậy rõ ràng, cơ chế fallback an toàn.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

# Lazy loader cho EasyOCR reader dùng chung
_SHARED_READER = None


def get_shared_easyocr_reader():
    """Lazy initialization cho EasyOCR reader để tránh nạp sớm khi unit test."""
    global _SHARED_READER
    if _SHARED_READER is None:
        try:
            from services.ocr_service import reader
            _SHARED_READER = reader
        except Exception as e:
            logger.warning(f"Không thể khởi tạo shared EasyOCR reader: {e}")
            _SHARED_READER = None
    return _SHARED_READER

logger = logging.getLogger(__name__)

# Từ khóa định danh quốc hiệu / tiêu đề mặt trước CCCD và GPLX
HEADER_KEYWORDS_CCCD = [
    "CONG HOA", "XA HOI", "CHU NGHIA", "VIET NAM", "DOC LAP", "TU DO", "HANH PHUC",
    "CAN CUOC", "CONG DAN", "CITIZEN", "IDENTITY", "CARD", "SO", "NO",
]

HEADER_KEYWORDS_GPLX = [
    "CONG HOA", "XA HOI", "CHU NGHIA", "VIET NAM", "DOC LAP", "TU DO", "HANH PHUC",
    "GIAY PHEP", "LAI XE", "DRIVING", "LICENCE", "LICENSE", "SO", "NO", "BO GIAO THONG",
]

CARD_TYPE_KEYWORDS = {
    "cccd": HEADER_KEYWORDS_CCCD,
    "gplx": HEADER_KEYWORDS_GPLX,
}


@dataclass
class OrientationResult:
    """Kết quả xác định và hiệu chỉnh chiều đọc của tài liệu."""

    success: bool
    image: np.ndarray  # Ảnh đã được xoay về hướng đọc chuẩn (Upright)
    rotation_deg: int  # Góc xoay đã áp dụng: 0, 90, 180, 270 (độ theo chiều kim đồng hồ)
    method: str  # 'anchor_header_ocr' | 'portrait_heuristic' | 'fallback_unmodified'
    confidence: float  # Điểm số tin cậy (0.0 -> 1.0)
    fallback_used: bool
    details: dict

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "rotation_deg": self.rotation_deg,
            "method": self.method,
            "confidence": round(self.confidence, 4),
            "fallback_used": self.fallback_used,
            "details": self.details,
            "image_shape": list(self.image.shape),
        }


def _strip_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt và chuẩn hóa chữ hoa để so khớp từ khóa."""
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return stripped.replace("Đ", "D").replace("đ", "d").upper()


class DocumentOrientationCorrector:
    """Bộ xác định và hiệu chỉnh chiều đọc văn bản cho giấy tờ tùy thân."""

    def __init__(
        self,
        ocr_reader=None,
        header_height_ratio: float = 0.28,
        min_match_confidence: float = 0.35,
    ) -> None:
        """Khởi tạo corrector.

        Args:
            ocr_reader: Instance EasyOCR Reader dùng chung để tránh nạp lại model.
            header_height_ratio: Tỷ lệ dải tiêu đề ở đầu thẻ cần quét (0.28 = 28% trên).
            min_match_confidence: Ngưỡng điểm tin cậy tối thiểu để chấp nhận góc xoay.
        """
        self._reader = ocr_reader
        self.header_height_ratio = header_height_ratio
        self.min_match_confidence = min_match_confidence

    @property
    def reader(self):
        if self._reader is None:
            self._reader = get_shared_easyocr_reader()
        return self._reader

    def correct_orientation(
        self,
        image: np.ndarray,
        card_type: str = "cccd",
        check_all_angles: bool = False,
    ) -> OrientationResult:
        """Xác định và xoay ảnh về đúng chiều đọc văn bản chuẩn.

        Args:
            image: Ảnh thẻ đã nắn phẳng (chuẩn 856x540 hoặc crop).
            card_type: 'cccd' hoặc 'gplx'.
            check_all_angles: True nếu cần quét cả 90° và 270°, False nếu chỉ quét 0° và 180°.

        Returns:
            OrientationResult chứa ảnh đúng chiều và góc xoay.
        """
        if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim < 2:
            return OrientationResult(
                success=False,
                image=image if isinstance(image, np.ndarray) else np.zeros((540, 856, 3), dtype=np.uint8),
                rotation_deg=0,
                method="invalid_input",
                confidence=0.0,
                fallback_used=True,
                details={"error": "Invalid image input"},
            )

        card_type = card_type.lower() if card_type in ("cccd", "gplx") else "cccd"
        h, w = image.shape[:2]

        # Nếu ảnh đang có chiều cao > chiều rộng (dọc) hoặc check_all_angles=True
        # thì bắt buộc phải kiểm tra cả 4 góc [0, 90, 180, 270]
        if h > w or check_all_angles:
            angles_to_test = [0, 90, 180, 270]
        else:
            # Thẻ đã được chuẩn hóa về landscape (ngang), chỉ cần kiểm tra 0° vs 180°
            angles_to_test = [0, 180]

        best_angle = 0
        best_score = -1.0
        best_details: dict = {}
        all_scores: dict[int, dict] = {}

        for angle in angles_to_test:
            rotated = self._rotate_image(image, angle)
            score, details = self._score_orientation(rotated, card_type)
            all_scores[angle] = {"score": round(score, 4), **details}

            if score > best_score:
                best_score = score
                best_angle = angle
                best_details = details

            # Nếu điểm số góc 0° đã rất xuất sắc (> 0.75), early exit để tiết kiệm thời gian CPU
            if angle == 0 and score >= 0.75:
                break

        # Đánh giá kết quả
        # Nếu điểm số tốt nhất đủ cao
        if best_score >= self.min_match_confidence:
            corrected_image = self._rotate_image(image, best_angle)
            return OrientationResult(
                success=True,
                image=corrected_image,
                rotation_deg=best_angle,
                method="anchor_header_ocr",
                confidence=min(1.0, float(best_score)),
                fallback_used=False,
                details={
                    "best_angle": best_angle,
                    "best_score": round(best_score, 4),
                    "all_scores": all_scores,
                    **best_details,
                },
            )

        # Fallback an toàn: Không đủ bằng chứng để khẳng định góc xoay
        logger.info(
            f"Orientation score thấp ({best_score:.3f} < {self.min_match_confidence}). "
            "Giữ nguyên góc 0° để bảo toàn dữ liệu gốc."
        )
        return OrientationResult(
            success=False,
            image=image.copy(),
            rotation_deg=0,
            method="fallback_unmodified",
            confidence=max(0.0, float(best_score)),
            fallback_used=True,
            details={
                "warning": "Low confidence orientation, keeping original image",
                "best_angle": best_angle,
                "best_score": round(best_score, 4),
                "all_scores": all_scores,
            },
        )

    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Xoay ảnh theo góc 0, 90, 180, 270 độ theo chiều kim đồng hồ."""
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            raise ValueError(f"Unsupported rotation angle: {angle}. Must be 0, 90, 180, or 270.")

    def _score_orientation(self, image: np.ndarray, card_type: str) -> tuple[float, dict]:
        """Chấm điểm mức độ phù hợp của ảnh ở góc hiện tại.

        Cắt dải tiêu đề trên đỉnh thẻ (Anchor Header Strip: top 28%) và chạy EasyOCR cục bộ.
        """
        h, w = image.shape[:2]
        header_h = max(20, int(h * self.header_height_ratio))
        header_crop = image[0:header_h, 0:w]

        # Chạy OCR nhanh trên dải tiêu đề
        # Sử dụng beamWidth=1 hoặc greedy search nhẹ để tối ưu CPU
        try:
            results = self.reader.readtext(
                header_crop,
                decoder="greedy",
                mag_ratio=1.0,
                contrast_ths=0.1,
                adjust_contrast=0.5,
            )
        except Exception as e:
            logger.warning(f"Lỗi khi OCR dải tiêu đề: {e}")
            return 0.0, {"error": str(e)}

        keywords = CARD_TYPE_KEYWORDS.get(card_type, HEADER_KEYWORDS_CCCD)
        matched_keywords: list[str] = []
        raw_texts: list[str] = []
        total_conf = 0.0
        num_boxes = 0

        for box, text, conf in results:
            if conf < 0.25:
                continue
            num_boxes += 1
            total_conf += conf
            norm_text = _strip_accents(text)
            raw_texts.append(norm_text)

            for kw in keywords:
                if kw in norm_text:
                    matched_keywords.append(kw)

        # Tính điểm số:
        # 1. Số từ khóa khớp (mỗi từ khóa khớp đóng góp 0.3 điểm)
        kw_score = min(0.80, len(set(matched_keywords)) * 0.25)

        # 2. Điểm tin cậy OCR của các khối văn bản tìm được
        avg_conf = (total_conf / num_boxes) if num_boxes > 0 else 0.0
        conf_score = min(0.20, avg_conf * 0.20)

        # 3. Phạt nếu không tìm thấy bất kỳ khối văn bản nào ở vùng trên
        density_penalty = 0.0 if num_boxes > 0 else -0.5

        final_score = max(0.0, kw_score + conf_score + density_penalty)

        details = {
            "matched_keywords": list(set(matched_keywords)),
            "num_text_boxes": num_boxes,
            "avg_ocr_conf": round(avg_conf, 3),
            "raw_texts_snippet": raw_texts[:5],
        }
        return final_score, details


# Instance mặc định dùng chung
default_orientation_corrector = DocumentOrientationCorrector()


def correct_orientation(
    rectified_image: np.ndarray,
    card_type: str = "cccd",
    check_all_angles: bool = False,
) -> OrientationResult:
    """Hàm tiện ích cấp module để hiệu chỉnh chiều đọc văn bản."""
    return default_orientation_corrector.correct_orientation(
        rectified_image, card_type=card_type, check_all_angles=check_all_angles
    )
