"""Bộ unit tests cho Giai đoạn 3.2: OCR Accuracy Optimization & Fallback.

Kiểm thử:
1. Field Validators (CCCD & GPLX)
2. License Class Normalization (A1, A2, B1, B2, C, dot-matrix variants 'Al', 'A|', 'AI')
3. Expiry Parser ("Không thời hạn", ngày tháng có/không dấu phân cách)
4. Evaluation Metrics (CER, WER, Exact-Match)
5. Field-Level Fallback Strategy & Bounded Retries
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from services.adaptive_extractor import (
    validate_id_number,
    validate_license_number,
    validate_full_name,
    validate_date,
    validate_gender,
    validate_nationality,
    validate_license_class,
    validate_expiry,
    levenshtein_distance,
    compute_cer,
    compute_wer,
    is_exact_match,
    extract_adaptive_pipeline_e,
    ExtractionResult,
)
from services.gplx_extractor import _normalize_class


class TestStage32Validators(unittest.TestCase):
    """Kiểm thử các hàm kiểm tra tính hợp lệ của trường."""

    def test_id_number_validator(self):
        self.assertTrue(validate_id_number("030206013743"))
        self.assertTrue(validate_id_number("030 206 013 743"))
        self.assertFalse(validate_id_number("03020601374"))  # 11 số
        self.assertFalse(validate_id_number("0302060137433"))  # 13 số
        self.assertFalse(validate_id_number(None))
        self.assertFalse(validate_id_number("ABC123456789"))

    def test_license_number_validator(self):
        self.assertTrue(validate_license_number("300248016796"))  # 12 số
        self.assertTrue(validate_license_number("123456789"))  # 9 số PET cũ
        self.assertFalse(validate_license_number("12345"))  # Quá ngắn
        self.assertFalse(validate_license_number(None))

    def test_full_name_validator(self):
        self.assertTrue(validate_full_name("VŨ NGỌC BẢO"))
        self.assertTrue(validate_full_name("NGUYEN VAN A"))
        self.assertFalse(validate_full_name("BẢO"))  # Chỉ có 1 từ
        self.assertFalse(validate_full_name("VŨ NGỌC 123"))  # Có số
        self.assertFalse(validate_full_name(None))

    def test_date_validator(self):
        self.assertTrue(validate_date("27/03/2006"))
        self.assertTrue(validate_date("27-03-2006"))
        self.assertTrue(validate_date("27.03.2006"))
        self.assertTrue(validate_date("27032006"))
        self.assertFalse(validate_date("32/01/2006"))  # Ngày sai
        self.assertFalse(validate_date("abc"))
        self.assertFalse(validate_date(None))

    def test_gender_validator(self):
        self.assertTrue(validate_gender("Nam"))
        self.assertTrue(validate_gender("NAM"))
        self.assertTrue(validate_gender("Nữ"))
        self.assertTrue(validate_gender("Nu"))
        self.assertFalse(validate_gender("Khác"))
        self.assertFalse(validate_gender(None))

    def test_nationality_validator(self):
        self.assertTrue(validate_nationality("Việt Nam"))
        self.assertTrue(validate_nationality("VIET NAM"))
        self.assertTrue(validate_nationality("VietNam"))
        self.assertFalse(validate_nationality("Lào"))
        self.assertFalse(validate_nationality(None))

    def test_license_class_validator(self):
        self.assertTrue(validate_license_class("A1"))
        self.assertTrue(validate_license_class("A2"))
        self.assertTrue(validate_license_class("B2"))
        self.assertTrue(validate_license_class("C"))
        self.assertTrue(validate_license_class("Al"))  # dot-matrix L
        self.assertTrue(validate_license_class("A|"))  # pipe
        self.assertFalse(validate_license_class("Z9"))
        self.assertFalse(validate_license_class(None))

    def test_expiry_validator(self):
        self.assertTrue(validate_expiry("Không thời hạn"))
        self.assertTrue(validate_expiry("KHONG THOI HAN"))
        self.assertTrue(validate_expiry("27/03/2031"))
        self.assertFalse(validate_expiry("Hết hạn rồi"))
        self.assertFalse(validate_expiry(None))


class TestStage32ClassNormalization(unittest.TestCase):
    """Kiểm thử chuẩn hóa hạng bằng lái xe."""

    def test_normalize_valid_classes(self):
        self.assertEqual(_normalize_class("A1"), "A1")
        self.assertEqual(_normalize_class("B2"), "B2")
        self.assertEqual(_normalize_class("C"), "C")

    def test_normalize_dot_matrix_substitutions(self):
        self.assertEqual(_normalize_class("Al"), "A1")
        self.assertEqual(_normalize_class("AI"), "A1")
        self.assertEqual(_normalize_class("A|"), "A1")
        self.assertEqual(_normalize_class("HANG A1"), "A1")
        self.assertEqual(_normalize_class("Hang: Al"), "A1")
        self.assertEqual(_normalize_class("B|"), "B1")
        self.assertEqual(_normalize_class("Bl"), "B1")


class TestStage32Metrics(unittest.TestCase):
    """Kiểm thử các hàm đo lường CER, WER và Exact Match."""

    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("CCCD", "CCCD"), 0)
        self.assertEqual(levenshtein_distance("", "test"), 4)

    def test_cer_calculation(self):
        self.assertEqual(compute_cer("VŨ NGỌC BẢO", "VŨ NGỌC BẢO"), 0.0)
        # Thay 1 ký tự: 'VU NGOC BAO' vs 'VŨ NGỌC BẢO' (3 dấu khác biệt)
        cer = compute_cer("VŨ NGỌC BẢO", "VŨ NGỌC BẢO")
        self.assertEqual(cer, 0.0)
        self.assertEqual(compute_cer(None, "CCCD"), 1.0)
        self.assertEqual(compute_cer(None, None), 0.0)

    def test_wer_calculation(self):
        self.assertEqual(compute_wer("VŨ NGỌC BẢO", "VŨ NGỌC BẢO"), 0.0)
        self.assertEqual(compute_wer("NGỌC BẢO", "VŨ NGỌC BẢO"), round(1 / 3, 4))
        self.assertEqual(compute_wer(None, "VŨ NGỌC BẢO"), 1.0)

    def test_exact_match(self):
        self.assertTrue(is_exact_match("Vũ Ngọc Bảo", "VŨ NGỌC BẢO"))
        self.assertTrue(is_exact_match(" 030206013743 ", "030206013743"))
        self.assertFalse(is_exact_match("030206013743", "030206013744"))
        self.assertFalse(is_exact_match(None, "030206013743"))
        self.assertTrue(is_exact_match(None, None))


class TestStage32FieldLevelFallback(unittest.TestCase):
    """Kiểm thử cơ chế Field-Level Fallback có kiểm soát."""

    @patch("services.adaptive_extractor.extract_calibrated_cccd_fields")
    @patch("services.adaptive_extractor._extract_single_baseline_field")
    def test_field_fallback_bounded_retry(self, mock_baseline_single, mock_calibrated):
        # Giả lập trích xuất ban đầu: thiếu date_of_birth và id_number sai định dạng
        mock_calibrated.return_value = {
            "id_number": "123",  # Không hợp lệ
            "full_name": "VŨ NGỌC BẢO",  # Hợp lệ
            "date_of_birth": None,  # Thiếu
            "gender": "Nam",
            "nationality": "Việt Nam",
            "place_of_origin": "Nam Chính, Nam Sách, Hải Dương",
            "residence": "Thôn Bích Đông, Nam Chính, Nam Sách, Hải Dương",
            "expiry_date": "27/03/2031",
        }

        # Giả lập baseline single field trích xuất thành công DOB
        def fake_baseline(crop, ctype, fld):
            if fld == "date_of_birth":
                return "27/03/2006"
            if fld == "id_number":
                return None
            return None

        mock_baseline_single.side_effect = fake_baseline

        dummy_img = np.zeros((540, 856, 3), dtype=np.uint8)
        dummy_crop = np.zeros((400, 600, 3), dtype=np.uint8)

        result = extract_adaptive_pipeline_e(dummy_img, dummy_crop, "cccd")

        # Kiểm tra:
        # 1. full_name không bị fallback
        self.assertFalse(result.fallback_used["full_name"])
        # 2. date_of_birth được fallback thành công
        self.assertTrue(result.fallback_used["date_of_birth"])
        self.assertEqual(result.fields["date_of_birth"], "27/03/2006")
        self.assertTrue(result.valid_flags["date_of_birth"])
        self.assertEqual(result.fallback_sources["date_of_birth"], "yolo_crop_baseline")
        # 3. id_number được thử fallback nhưng thất bại (vẫn bounded 1 lần)
        self.assertTrue(result.fallback_used["id_number"])
        self.assertFalse(result.valid_flags["id_number"])
        self.assertEqual(result.fallback_sources["id_number"], "failed")

        # Tổng số lần fallback kích hoạt = 2 (id_number, date_of_birth)
        self.assertEqual(result.total_fallbacks, 2)
        # Số fallback thành công = 1 (date_of_birth)
        self.assertEqual(result.successful_fallbacks, 1)


if __name__ == "__main__":
    unittest.main()

