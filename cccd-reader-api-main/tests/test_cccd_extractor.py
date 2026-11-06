"""Unit test cho các hàm mới trong cccd_extractor.py.

Chạy (từ thư mục cccd-reader-api-main):
    python -m unittest tests.test_cccd_extractor -v

Không cần ảnh thật và không cần EasyOCR / cv2 đầy đủ được cài sẵn.
Mọi phụ thuộc I/O và xử lý ảnh được mock — chỉ kiểm tra logic text.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub các module nặng TRƯỚC KHI bất kỳ services nào được import.
# ocr_service.py khởi tạo easyocr.Reader() ở module level nên phải stub
# toàn bộ module services.ocr_service thay vì chỉ stub easyocr.
# ---------------------------------------------------------------------------
_ocr_service_stub = types.ModuleType("services.ocr_service")
_ocr_service_stub.read_text = MagicMock(return_value=[])
sys.modules["services.ocr_service"] = _ocr_service_stub

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402  (import sau khi stub)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_id_number(ocr_parts: list[str]) -> str | None:
    """Chạy extract_id_number với _ocr_region được mock."""
    from services.cccd_extractor import extract_id_number
    with patch("services.cccd_extractor._ocr_region", return_value=ocr_parts):
        return extract_id_number(MagicMock())


def _run_expiry(ocr_parts: list[str]) -> str | None:
    """Chạy extract_expiry_date với _ocr_region được mock."""
    from services.cccd_extractor import extract_expiry_date
    with patch("services.cccd_extractor._ocr_region", return_value=ocr_parts):
        return extract_expiry_date(MagicMock())


# ---------------------------------------------------------------------------
# Tests: extract_id_number
# ---------------------------------------------------------------------------

class TestExtractIdNumber(unittest.TestCase):

    def test_valid_12_digits(self):
        self.assertEqual(_run_id_number(["079123456789"]), "079123456789")

    def test_ocr_error_O_as_zero(self):
        self.assertEqual(_run_id_number(["O79123456789"]), "079123456789")

    def test_ocr_error_I_as_one(self):
        self.assertEqual(_run_id_number(["079I23456789"]), "079123456789")

    def test_ocr_error_l_as_one(self):
        self.assertEqual(_run_id_number(["0791234567l9"]), "079123456719")

    def test_ocr_error_S_as_five(self):
        self.assertEqual(_run_id_number(["07S123456789"]), "075123456789")

    def test_ocr_error_B_as_eight(self):
        self.assertEqual(_run_id_number(["079123456B89"]), "079123456889")

    def test_digits_with_spaces(self):
        self.assertEqual(_run_id_number(["079 123 456 789"]), "079123456789")

    def test_digits_with_hyphens(self):
        self.assertEqual(_run_id_number(["079-123-456-789"]), "079123456789")

    def test_multipart_ocr(self):
        """OCR trả nhiều phần tử → ghép lại."""
        self.assertEqual(_run_id_number(["079123", "456789"]), "079123456789")

    def test_fewer_than_12_returns_none(self):
        self.assertIsNone(_run_id_number(["07912345678"]))  # 11 chữ số

    def test_13_digits_takes_first_12(self):
        self.assertEqual(_run_id_number(["0791234567890"]), "079123456789")

    def test_no_ocr_parts(self):
        self.assertIsNone(_run_id_number([]))

    def test_non_digit_text_returns_none(self):
        self.assertIsNone(_run_id_number(["ABCDEFGHIJKL"]))

    def test_does_not_fabricate_missing_digits(self):
        self.assertIsNone(_run_id_number(["0791234567"]))  # 10 chữ số


# ---------------------------------------------------------------------------
# Tests: extract_expiry_date
# ---------------------------------------------------------------------------

class TestExtractExpiryDate(unittest.TestCase):

    def test_slash_format(self):
        self.assertEqual(_run_expiry(["12/08/2031"]), "12/08/2031")

    def test_dot_format_normalized(self):
        self.assertEqual(_run_expiry(["12.08.2031"]), "12/08/2031")

    def test_hyphen_format_normalized(self):
        self.assertEqual(_run_expiry(["12-08-2031"]), "12/08/2031")

    def test_spaces_between_parts(self):
        self.assertEqual(_run_expiry(["12 / 08 / 2031"]), "12/08/2031")

    def test_invalid_month_13_returns_none(self):
        self.assertIsNone(_run_expiry(["01/13/2031"]))

    def test_invalid_day_32_returns_none(self):
        self.assertIsNone(_run_expiry(["32/01/2031"]))

    def test_no_date_pattern_returns_none(self):
        self.assertIsNone(_run_expiry(["Khong co ngay thang"]))

    def test_no_ocr_parts(self):
        self.assertIsNone(_run_expiry([]))

    def test_valid_leap_year(self):
        self.assertEqual(_run_expiry(["29/02/2032"]), "29/02/2032")

    def test_invalid_leap_year(self):
        self.assertIsNone(_run_expiry(["29/02/2031"]))

    def test_date_embedded_in_text(self):
        self.assertEqual(_run_expiry(["Co gia tri den: 12/08/2031 (het han)"]), "12/08/2031")

    def test_does_not_fabricate_date(self):
        self.assertIsNone(_run_expiry([]))


# ---------------------------------------------------------------------------
# Tests: schema output extract_cccd_fields
# ---------------------------------------------------------------------------

class TestExtractCccdFieldsSchema(unittest.TestCase):

    EXPECTED_KEYS = [
        "id_number", "full_name", "date_of_birth", "gender",
        "nationality", "place_of_origin", "residence", "expiry_date",
    ]

    def _run(self):
        from services.cccd_extractor import extract_cccd_fields
        with (
            patch("services.cccd_extractor.cv2") as mock_cv2,
            patch("services.cccd_extractor._ocr_region", return_value=[]),
            patch("services.cccd_extractor._normalize_width", return_value=MagicMock()),
        ):
            mock_cv2.imread.return_value = MagicMock()
            return extract_cccd_fields("fake.jpg")

    def test_output_contains_new_fields(self):
        result = self._run()
        for key in self.EXPECTED_KEYS:
            self.assertIn(key, result, f"Thiếu key '{key}'")

    def test_all_values_none_on_no_ocr(self):
        result = self._run()
        for key, val in result.items():
            self.assertIsNone(val, f"Key '{key}' phải là None")

    def test_id_number_is_first_key(self):
        result = self._run()
        self.assertEqual(list(result.keys())[0], "id_number")

    def test_expiry_date_is_last_key(self):
        result = self._run()
        self.assertEqual(list(result.keys())[-1], "expiry_date")


# ---------------------------------------------------------------------------
# Tests: GPLX không bị ảnh hưởng
# ---------------------------------------------------------------------------

class TestGplxExtractorUnchanged(unittest.TestCase):

    EXPECTED_GPLX_KEYS = [
        "license_number", "full_name", "date_of_birth",
        "nationality", "address", "license_class", "expiry",
    ]

    def test_gplx_extractor_import(self):
        from services import gplx_extractor  # noqa: F401

    def test_gplx_has_all_expected_keys(self):
        from services.gplx_extractor import extract_gplx_fields
        # gplx_extractor gọi _ocr_region qua cccd_extractor → mock đúng symbol
        fake_image = MagicMock()
        fake_image.shape = (670, 1061, 3)  # shape hợp lệ để _crop_region không lỗi
        with (
            patch("services.gplx_extractor.cv2") as mock_cv2,
            patch("services.gplx_extractor._ocr_region", return_value=[]),
            patch("services.gplx_extractor._normalize_width", return_value=fake_image),
        ):
            mock_cv2.imread.return_value = fake_image
            result = extract_gplx_fields("fake.jpg")
        for key in self.EXPECTED_GPLX_KEYS:
            self.assertIn(key, result, f"gplx_extractor thiếu key '{key}'")

    def test_gplx_has_no_cccd_only_keys(self):
        """GPLX không có 'id_number' hay 'expiry_date' (đây là key chỉ có trong CCCD)."""
        from services.gplx_extractor import extract_gplx_fields
        fake_image = MagicMock()
        fake_image.shape = (670, 1061, 3)
        with (
            patch("services.gplx_extractor.cv2") as mock_cv2,
            patch("services.gplx_extractor._ocr_region", return_value=[]),
            patch("services.gplx_extractor._normalize_width", return_value=fake_image),
        ):
            mock_cv2.imread.return_value = fake_image
            result = extract_gplx_fields("fake.jpg")
        self.assertNotIn("id_number", result)
        self.assertNotIn("expiry_date", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
