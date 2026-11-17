"""Regression Tests cho Field Cleaner & Validation Semantics (Nhiệm vụ 6 - P1 Fix).

Bao gồm 12 kịch bản thử nghiệm sử dụng dữ liệu tổng hợp (Synthetic Data):
1. Tên có nhãn tiếng Anh đứng trước.
2. Tên có dấu tiếng Việt.
3. Tên dài nhiều từ.
4. Chuỗi 'LEN / NAMNE ...'.
5. Quê quán chứa 'Place oÍ origin'.
6. Thường trú chứa 'Placo of residence'.
7. Nhãn và giá trị nằm cùng một dòng.
8. Nhãn và giá trị nằm trên hai dòng.
9. Địa chỉ nhiều dòng.
10. Trường rỗng.
11. Chuỗi sạch không bị thay đổi.
12. Tránh xóa nhầm các từ hợp lệ trong tên/địa chỉ.

Tuân thủ nghiêm ngặt: Tuyệt đối không chứa PII thật của công dân trong code test.
"""

from __future__ import annotations

import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import unittest
from dataclasses import dataclass

from services.field_cleaner import (
    clean_full_name,
    clean_place_of_origin,
    clean_residence,
    has_label_contamination,
)
from services.adaptive_extractor import (
    validate_full_name,
    validate_place_of_origin,
    validate_residence,
    validate_id_number,
)


@dataclass
class MockOcrLine:
    text: str
    y_min: float = 0.0
    y_max: float = 0.0
    confidence: float = 0.95


class TestFieldCleanerSynthetic(unittest.TestCase):
    """Bộ kiểm thử hồi quy 12 kịch bản tổng hợp theo yêu cầu Nhiệm vụ 6."""

    # -----------------------------------------------------------------------
    # 1. Tên có nhãn tiếng Anh đứng trước
    # -----------------------------------------------------------------------
    def test_01_name_with_english_label_prefix(self):
        raw = "Full name NGUYỄN VĂN AN"
        cleaned, raw_out, is_clean = clean_full_name(raw)
        self.assertEqual(cleaned, "NGUYỄN VĂN AN")
        self.assertEqual(raw_out, raw)
        self.assertTrue(is_clean)
        self.assertTrue(validate_full_name(cleaned))

    # -----------------------------------------------------------------------
    # 2. Tên có dấu tiếng Việt
    # -----------------------------------------------------------------------
    def test_02_name_with_vietnamese_diacritics(self):
        raw = "VŨ NGỌC BẢO"
        cleaned, raw_out, is_clean = clean_full_name(raw)
        self.assertEqual(cleaned, "VŨ NGỌC BẢO")
        self.assertTrue(is_clean)
        self.assertTrue(validate_full_name(cleaned))

        raw_female = "ĐẶNG THỊ KIM NGÂN"
        cleaned_f, _, is_clean_f = clean_full_name(raw_female)
        self.assertEqual(cleaned_f, "ĐẶNG THỊ KIM NGÂN")
        self.assertTrue(is_clean_f)
        self.assertTrue(validate_full_name(cleaned_f))

    # -----------------------------------------------------------------------
    # 3. Tên dài nhiều từ
    # -----------------------------------------------------------------------
    def test_03_name_long_multi_word(self):
        raw = "NGUYỄN HOÀNG GIA BẢO PHÚC THỊNH"
        cleaned, _, is_clean = clean_full_name(raw)
        self.assertEqual(cleaned, "NGUYỄN HOÀNG GIA BẢO PHÚC THỊNH")
        self.assertTrue(is_clean)
        self.assertTrue(validate_full_name(cleaned))

    # -----------------------------------------------------------------------
    # 4. Chuỗi 'LEN / NAMNE ...'
    # -----------------------------------------------------------------------
    def test_04_name_with_len_namne_corruption(self):
        raw = "LEN / NAMNE VŨ NGOC BẢO"
        # Trước khi làm sạch, validator PHẢI trả về False
        self.assertFalse(validate_full_name(raw))
        self.assertTrue(has_label_contamination("full_name", raw))

        cleaned, raw_out, is_clean = clean_full_name(raw)
        self.assertEqual(cleaned, "VŨ NGOC BẢO")
        self.assertEqual(raw_out, raw)
        self.assertTrue(is_clean)
        # Sau khi làm sạch, validator PHẢI trả về True
        self.assertTrue(validate_full_name(cleaned))
        self.assertFalse(has_label_contamination("full_name", cleaned))

    # -----------------------------------------------------------------------
    # 5. Quê quán chứa 'Place oÍ origin'
    # -----------------------------------------------------------------------
    def test_05_origin_with_place_oi_origin(self):
        raw = "Que 1 Place oÍ origin Nam Chinh, Nam Sách Hải Dương quan"
        # Trước khi làm sạch, validator PHẢI trả về False
        self.assertFalse(validate_place_of_origin(raw))
        self.assertTrue(has_label_contamination("place_of_origin", raw))

        cleaned, raw_out, is_clean = clean_place_of_origin(raw)
        self.assertEqual(cleaned, "Nam Chinh, Nam Sách Hải Dương")
        self.assertEqual(raw_out, raw)
        self.assertTrue(is_clean)
        self.assertTrue(validate_place_of_origin(cleaned))

    # -----------------------------------------------------------------------
    # 6. Thường trú chứa 'Placo of residence'
    # -----------------------------------------------------------------------
    def test_06_residence_with_placo_of_residence(self):
        raw = "J trú / Placo of residence Thôn Bi Đông, L Nam Chinh Nam Sách, Hải Dương"
        # Trước khi làm sạch, validator PHẢI trả về False
        self.assertFalse(validate_residence(raw))
        self.assertTrue(has_label_contamination("residence", raw))

        cleaned, raw_out, is_clean = clean_residence(raw)
        self.assertEqual(cleaned, "Thôn Bi Đông, Nam Chinh Nam Sách, Hải Dương")
        self.assertEqual(raw_out, raw)
        self.assertTrue(is_clean)
        self.assertTrue(validate_residence(cleaned))

    # -----------------------------------------------------------------------
    # 7. Nhãn và giá trị nằm cùng một dòng
    # -----------------------------------------------------------------------
    def test_07_label_and_value_on_same_line(self):
        raw_name = "Họ và tên: TRẦN VĂN MINH"
        cleaned_name, _, is_clean_name = clean_full_name(raw_name)
        self.assertEqual(cleaned_name, "TRẦN VĂN MINH")
        self.assertTrue(is_clean_name)
        self.assertTrue(validate_full_name(cleaned_name))

        raw_orig = "Quê quán: Xã Tân Bình, Huyện Vĩnh Cửu, Đồng Nai"
        cleaned_orig, _, is_clean_orig = clean_place_of_origin(raw_orig)
        self.assertEqual(cleaned_orig, "Xã Tân Bình, Huyện Vĩnh Cửu, Đồng Nai")
        self.assertTrue(is_clean_orig)
        self.assertTrue(validate_place_of_origin(cleaned_orig))

    # -----------------------------------------------------------------------
    # 8. Nhãn và giá trị nằm trên hai dòng
    # -----------------------------------------------------------------------
    def test_08_label_and_value_on_two_lines(self):
        lines = [
            MockOcrLine(text="Họ và tên / Full name", y_min=0.45, y_max=0.50),
            MockOcrLine(text="TRẦN THỊ HỒNG", y_min=0.52, y_max=0.58),
        ]
        raw_combined = "Họ và tên / Full name TRẦN THỊ HỒNG"
        cleaned, raw_out, is_clean = clean_full_name(raw_combined, lines=lines)
        self.assertEqual(cleaned, "TRẦN THỊ HỒNG")
        self.assertTrue(is_clean)
        self.assertTrue(validate_full_name(cleaned))

    # -----------------------------------------------------------------------
    # 9. Địa chỉ nhiều dòng
    # -----------------------------------------------------------------------
    def test_09_multi_line_address(self):
        lines = [
            MockOcrLine(text="Số 123 Đường Nguyễn Trãi", y_min=0.81, y_max=0.86),
            MockOcrLine(text="Phường 2, Quận 5, TP. Hồ Chí Minh", y_min=0.88, y_max=0.94),
        ]
        raw_combined = "Số 123 Đường Nguyễn Trãi, Phường 2, Quận 5, TP. Hồ Chí Minh"
        cleaned, raw_out, is_clean = clean_residence(raw_combined, lines=lines)
        self.assertEqual(cleaned, "Số 123 Đường Nguyễn Trãi, Phường 2, Quận 5, TP. Hồ Chí Minh")
        self.assertTrue(is_clean)
        self.assertTrue(validate_residence(cleaned))

    # -----------------------------------------------------------------------
    # 10. Trường rỗng
    # -----------------------------------------------------------------------
    def test_10_empty_fields(self):
        for empty_val in [None, "", "   "]:
            c_name, r_name, ok_name = clean_full_name(empty_val)
            self.assertIsNone(c_name)
            self.assertFalse(validate_full_name(c_name))

            c_orig, r_orig, ok_orig = clean_place_of_origin(empty_val)
            self.assertIsNone(c_orig)
            self.assertFalse(validate_place_of_origin(c_orig))

            c_res, r_res, ok_res = clean_residence(empty_val)
            self.assertIsNone(c_res)
            self.assertFalse(validate_residence(c_res))

    # -----------------------------------------------------------------------
    # 11. Chuỗi sạch không bị thay đổi
    # -----------------------------------------------------------------------
    def test_11_clean_strings_unaltered(self):
        clean_name = "PHẠM QUANG HUY"
        c_name, _, ok_name = clean_full_name(clean_name)
        self.assertEqual(c_name, clean_name)
        self.assertTrue(ok_name)

        clean_orig = "Xã Đông Kết, Huyện Khoái Châu, Hưng Yên"
        c_orig, _, ok_orig = clean_place_of_origin(clean_orig)
        self.assertEqual(c_orig, clean_orig)
        self.assertTrue(ok_orig)

        clean_res = "Thôn 3, Xã Quảng Tâm, Huyện Tuy Đức, Đắk Nông"
        c_res, _, ok_res = clean_residence(clean_res)
        self.assertEqual(c_res, clean_res)
        self.assertTrue(ok_res)

    # -----------------------------------------------------------------------
    # 12. Tránh xóa nhầm các từ hợp lệ trong tên / địa chỉ
    # -----------------------------------------------------------------------
    def test_12_preserve_legitimate_tokens(self):
        # Tên người chứa các từ 'HỒ' (họ Hồ) hoặc 'NAM' (tên Nam)
        name_with_ho_nam = "HỒ VĂN NAM"
        c1, _, ok1 = clean_full_name(name_with_ho_nam)
        self.assertEqual(c1, "HỒ VĂN NAM")
        self.assertTrue(ok1)
        self.assertTrue(validate_full_name(c1))

        # Tên người chứa từ 'QUÊ'
        name_with_que = "ĐỖ VĂN QUÊ"
        c2, _, ok2 = clean_full_name(name_with_que)
        self.assertEqual(c2, "ĐỖ VĂN QUÊ")
        self.assertTrue(ok2)
        self.assertTrue(validate_full_name(c2))

        # Địa danh chứa chữ 'Quê' hoặc 'Quán' hợp lệ
        place_with_que = "Xã Quê Mỹ, Huyện Thăng Bình, Quảng Nam"
        c3, _, ok3 = clean_place_of_origin(place_with_que)
        self.assertEqual(c3, "Xã Quê Mỹ, Huyện Thăng Bình, Quảng Nam")
        self.assertTrue(ok3)
        self.assertTrue(validate_place_of_origin(c3))

        # Địa danh chứa chữ 'Quán'
        place_with_quan = "Thị trấn Quán Hành, Huyện Nghi Lộc, Nghệ An"
        c4, _, ok4 = clean_place_of_origin(place_with_quan)
        self.assertEqual(c4, "Thị trấn Quán Hành, Huyện Nghi Lộc, Nghệ An")
        self.assertTrue(ok4)
        self.assertTrue(validate_place_of_origin(c4))


if __name__ == "__main__":
    unittest.main()
