"""Bộ Unit Test toàn diện cho cccd_extractor và gplx_extractor (Giai đoạn 1).

Chạy kiểm thử:
    python -m unittest tests.test_cccd_extractor -v

Bao phủ các ca kiểm thử:
1. Số CCCD:
   - Đúng chuẩn 12 chữ số.
   - Bắt đầu bằng chữ số 0 (không bị mất dạng string).
   - Có 11 chữ số -> trả về None (không tự đoán).
   - Có 13 chữ số -> trả về None (không cắt xén).
   - Ký tự phân cách (khoảng trắng, gạch ngang, chấm...).
   - Sửa lỗi OCR nhầm O/0, I/1, l/1, S/5, B/8... trên chuỗi số hợp lệ.
   - Chuỗi ký tự họ tên không bị ép thành số.
   - Ghép nhiều phần tử số rời rạc trong cùng dòng.
   - Chống ghép các đoạn văn bản khác trường dẫn đến tạo số giả.
2. Thời hạn sử dụng (Có giá trị đến):
   - Ngày DD/MM/YYYY hợp lệ.
   - Chuẩn hóa phân cách chấm, gạch ngang thành gạch chéo /.
   - Ngày không hợp lệ trên lịch (tháng 13, ngày 32...).
   - Năm nhuận: 29/02 năm nhuận hợp lệ, năm không nhuận trả None.
   - Nhận diện nhãn "Không thời hạn".
   - Không tự tính thời hạn sử dụng.
   - Vùng ROI không có dữ liệu -> trả về None.
3. Toàn vẹn Schema & Hồi quy:
   - Đầy đủ 8 trường của CCCD theo thứ tự chuẩn.
   - Tất cả giá trị None khi ảnh không có chữ.
   - Hồi quy GPLX: giữ nguyên 7 trường gốc, không lẫn trường của CCCD.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Stub services.ocr_service trước khi import services
_ocr_stub = types.ModuleType("services.ocr_service")
_ocr_stub.read_text = MagicMock(return_value=[])
sys.modules["services.ocr_service"] = _ocr_stub

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _run_id_number(ocr_parts: list[str]) -> str | None:
    from services.cccd_extractor import extract_id_number
    with patch("services.cccd_extractor._ocr_region", return_value=ocr_parts):
        return extract_id_number(MagicMock())


def _run_expiry(ocr_parts: list[str]) -> str | None:
    from services.cccd_extractor import extract_expiry_date
    with patch("services.cccd_extractor._ocr_region", return_value=ocr_parts):
        return extract_expiry_date(MagicMock())


class TestExtractIdNumber(unittest.TestCase):
    """Kiểm tra logic trích xuất Số CCCD."""

    def test_valid_12_digits(self):
        """Số CCCD chuẩn 12 chữ số."""
        self.assertEqual(_run_id_number(["079123456789"]), "079123456789")

    def test_leading_zero_preserved(self):
        """Số CCCD bắt đầu bằng số 0 (ví dụ mã Hà Nội 001) phải giữ nguyên dạng chuỗi."""
        res = _run_id_number(["001202012345"])
        self.assertEqual(res, "001202012345")
        self.assertTrue(isinstance(res, str))
        self.assertTrue(res.startswith("001"))

    def test_eleven_digits_returns_none(self):
        """Thiếu 1 chữ số (11 số) -> Bắt buộc trả None, không đoán."""
        self.assertIsNone(_run_id_number(["07912345678"]))

    def test_thirteen_digits_returns_none(self):
        """Thừa 1 chữ số (13 số) -> Bắt buộc trả None, KHÔNG cắt xén 12 số đầu."""
        self.assertIsNone(_run_id_number(["0791234567890"]))

    def test_ten_digits_returns_none(self):
        """10 chữ số -> None."""
        self.assertIsNone(_run_id_number(["0123456789"]))

    def test_separators_spaces(self):
        """Chuỗi có khoảng trắng giữa các cụm số."""
        self.assertEqual(_run_id_number(["079 123 456 789"]), "079123456789")

    def test_separators_hyphens_dots(self):
        """Chuỗi có gạch ngang hoặc dấu chấm."""
        self.assertEqual(_run_id_number(["079-123-456-789"]), "079123456789")
        self.assertEqual(_run_id_number(["079.123.456.789"]), "079123456789")

    def test_ocr_fix_O_to_0(self):
        """Sửa lỗi chữ O hoa thành số 0."""
        self.assertEqual(_run_id_number(["O79123456789"]), "079123456789")

    def test_ocr_fix_I_and_l_to_1(self):
        """Sửa lỗi chữ I hoa hoặc l thường thành số 1."""
        self.assertEqual(_run_id_number(["079I23456789"]), "079123456789")
        self.assertEqual(_run_id_number(["0791234567l9"]), "079123456719")

    def test_ocr_fix_S_and_B(self):
        """Sửa lỗi chữ S thành 5 và B thành 8."""
        self.assertEqual(_run_id_number(["07S123456789"]), "075123456789")
        self.assertEqual(_run_id_number(["079123456B89"]), "079123456889")

    def test_text_with_names_not_corrupted(self):
        """Văn bản họ tên hoặc từ ngữ không bị sửa thành số linh tinh."""
        self.assertIsNone(_run_id_number(["NGUYEN VAN A"]))
        self.assertIsNone(_run_id_number(["CONG HOA XA HOI"]))

    def test_multiple_parts_combining_valid_digits(self):
        """Các cụm số bị tách thành nhiều token nhưng tổng đúng 12 chữ số."""
        self.assertEqual(_run_id_number(["079", "123", "456789"]), "079123456789")

    def test_multiple_fields_confusion_prevents_false_combination(self):
        """Không ghép văn bản lộn xộn từ trường khác tạo thành 12 số."""
        # Ví dụ có text chữ 'So' và số ngắn '079'
        self.assertIsNone(_run_id_number(["So:", "Nam", "079123"]))

    def test_empty_ocr_returns_none(self):
        """Không có kết quả OCR -> trả None."""
        self.assertIsNone(_run_id_number([]))


class TestExtractExpiryDate(unittest.TestCase):
    """Kiểm tra logic trích xuất Thời hạn sử dụng (Có giá trị đến)."""

    def test_valid_expiry_slash(self):
        """Định dạng DD/MM/YYYY chuẩn."""
        self.assertEqual(_run_expiry(["15/10/2030"]), "15/10/2030")

    def test_valid_expiry_dot_and_hyphen(self):
        """Định dạng dấu chấm hoặc gạch ngang được chuẩn hóa thành /."""
        self.assertEqual(_run_expiry(["15.10.2030"]), "15/10/2030")
        self.assertEqual(_run_expiry(["15-10-2030"]), "15/10/2030")

    def test_valid_expiry_embedded_in_label(self):
        """Ngày nằm cùng dòng với nhãn tiếng Việt/Anh."""
        self.assertEqual(
            _run_expiry(["Có giá trị đến / Date of expiry: 24/05/2035"]),
            "24/05/2035",
        )

    def test_invalid_month_returns_none(self):
        """Tháng 13 không tồn tại -> None."""
        self.assertIsNone(_run_expiry(["15/13/2030"]))

    def test_invalid_day_returns_none(self):
        """Ngày 32 không tồn tại -> None."""
        self.assertIsNone(_run_expiry(["32/05/2030"]))

    def test_leap_year_valid(self):
        """Năm nhuận 2032 có ngày 29/02 hợp lệ."""
        self.assertEqual(_run_expiry(["29/02/2032"]), "29/02/2032")

    def test_non_leap_year_invalid(self):
        """Năm không nhuận 2031 không có ngày 29/02 -> None."""
        self.assertIsNone(_run_expiry(["29/02/2031"]))

    def test_no_expiry_limit_label(self):
        """Trường hợp 'Không thời hạn'."""
        self.assertEqual(_run_expiry(["Không thời hạn"]), "Không thời hạn")

    def test_empty_returns_none(self):
        """Vùng không đọc được -> None."""
        self.assertIsNone(_run_expiry([]))

    def test_random_text_without_date_returns_none(self):
        """Không chứa ngày tháng -> None."""
        self.assertIsNone(_run_expiry(["Quoc gia Viet Nam"]))


class TestCccdSchemaAndRegression(unittest.TestCase):
    """Kiểm tra toàn vẹn schema JSON và hồi quy GPLX."""

    EXPECTED_CCCD_KEYS = [
        "id_number",
        "full_name",
        "date_of_birth",
        "gender",
        "nationality",
        "place_of_origin",
        "residence",
        "expiry_date",
    ]

    EXPECTED_GPLX_KEYS = [
        "license_number",
        "full_name",
        "date_of_birth",
        "nationality",
        "address",
        "license_class",
        "expiry",
    ]

    def test_cccd_full_schema(self):
        """extract_cccd_fields trả đúng 8 trường theo đúng thứ tự."""
        from services.cccd_extractor import extract_cccd_fields

        fake_img = MagicMock()
        fake_img.shape = (670, 1061, 3)

        with (
            patch("services.cccd_extractor.cv2") as mock_cv2,
            patch("services.cccd_extractor._ocr_region", return_value=[]),
            patch("services.cccd_extractor._normalize_width", return_value=fake_img),
        ):
            mock_cv2.imread.return_value = fake_img
            result = extract_cccd_fields("dummy.jpg")

        self.assertEqual(list(result.keys()), self.EXPECTED_CCCD_KEYS)
        for key in self.EXPECTED_CCCD_KEYS:
            self.assertIsNone(result[key])

    def test_gplx_regression_schema(self):
        """extract_gplx_fields giữ nguyên 7 trường, không dính trường của CCCD."""
        from services.gplx_extractor import extract_gplx_fields

        fake_img = MagicMock()
        fake_img.shape = (670, 1061, 3)

        with (
            patch("services.gplx_extractor.cv2") as mock_cv2,
            patch("services.gplx_extractor._ocr_region", return_value=[]),
            patch("services.gplx_extractor._normalize_width", return_value=fake_img),
        ):
            mock_cv2.imread.return_value = fake_img
            result = extract_gplx_fields("dummy_gplx.jpg")

        self.assertEqual(list(result.keys()), self.EXPECTED_GPLX_KEYS)
        self.assertNotIn("id_number", result)
        self.assertNotIn("expiry_date", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
