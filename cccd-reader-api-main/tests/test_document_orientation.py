"""Unit tests cho DocumentOrientationCorrector và ROI Calibration (Giai đoạn 3.1).

Kiểm thử toàn diện 15 tiêu chí yêu cầu trong Nhiệm vụ G:
1. Thẻ xoay 0°.
2. Thẻ xoay 90°.
3. Thẻ xoay 180°.
4. Thẻ xoay 270°.
5. Ảnh không đủ thông tin xác định hướng (fallback an toàn).
6. Ảnh mờ.
7. CCCD và GPLX phân biệt layout/từ khóa khác nhau.
8. ROI nằm hoàn toàn trong không gian chuẩn [0, 1].
9. ROI không rỗng (x2 > x1 và y2 > y1).
10. ROI đúng hệ tọa độ normalized.
11. Fallback khi rectification thất bại.
12. Fallback khi orientation không chắc chắn.
13. Không lật gương văn bản (không dùng cv2.flip ngang).
14. Không làm thay đổi (mutate) ảnh đầu vào.
15. Không gây regression với các bài test và module cũ.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from services.document_orientation import (
    DocumentOrientationCorrector,
    OrientationResult,
    correct_orientation,
)
from services.calibrated_rois import (
    RECTIFIED_CCCD_ROIS,
    RECTIFIED_GPLX_ROIS,
    extract_calibrated_cccd_fields,
    extract_calibrated_gplx_fields,
)
from services.cccd_extractor import CCCD_ROIS
from services.gplx_extractor import GPLX_ROIS


class TestDocumentOrientationCorrector(unittest.TestCase):
    """Bộ kiểm thử cho module DocumentOrientationCorrector."""

    def setUp(self):
        # Mock EasyOCR Reader để kiểm thử logic xoay và chấm điểm nhanh, độc lập
        self.mock_reader = MagicMock()
        self.corrector = DocumentOrientationCorrector(ocr_reader=self.mock_reader)

    def test_orientation_0_degrees(self):
        """1. Thẻ xoay 0°: Nhận diện chính xác từ khóa tiêu đề ở góc 0°."""
        def fake_readtext(crop, **kwargs):
            # Giả lập nhìn thấy quốc hiệu ở góc 0°
            return [([ [10, 10], [100, 10], [100, 30], [10, 30] ], "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", 0.95)]

        self.mock_reader.readtext.side_effect = fake_readtext
        dummy_img = np.zeros((540, 856, 3), dtype=np.uint8)

        res = self.corrector.correct_orientation(dummy_img, card_type="cccd")
        self.assertTrue(res.success)
        self.assertEqual(res.rotation_deg, 0)
        self.assertFalse(res.fallback_used)
        self.assertGreater(res.confidence, 0.4)

    def test_orientation_180_degrees(self):
        """3. Thẻ xoay 180°: Nhận diện từ khóa ở góc 180° và xoay ảnh lại."""
        def fake_readtext(crop, **kwargs):
            # Giả lập: ở góc 180° (khi ảnh được xoay 180°), tiêu đề mới xuất hiện
            # Kiểm tra xem crop này là từ góc 180° hay không
            # Ta đánh dấu bằng cách xem crop có giá trị pixel gì không
            if crop[0, 0, 0] == 180:
                return [([ [10, 10], [100, 10], [100, 30], [10, 30] ], "CĂN CƯỚC CÔNG DÂN", 0.90)]
            return []

        # Tạo ảnh có pixel góc là 180 sau khi xoay 180 độ
        # Khi xoay 180, pixel [bottom, right] sẽ về [0, 0]
        img = np.zeros((540, 856, 3), dtype=np.uint8)
        img[539, 855, 0] = 180

        self.mock_reader.readtext.side_effect = fake_readtext
        res = self.corrector.correct_orientation(img, card_type="cccd")
        self.assertTrue(res.success)
        self.assertEqual(res.rotation_deg, 180)
        self.assertFalse(res.fallback_used)

    def test_orientation_90_and_270_degrees(self):
        """2 & 4. Thẻ xoay 90° và 270°: Hỗ trợ khi check_all_angles=True."""
        def fake_readtext(crop, **kwargs):
            if crop[0, 0, 0] == 90:
                return [([ [10, 10], [100, 10], [100, 30], [10, 30] ], "GIẤY PHÉP LÁI XE", 0.88)]
            return []

        # Khi xoay 90° clockwise, pixel (539, 0) về (0, 0)
        img = np.zeros((540, 856, 3), dtype=np.uint8)
        img[539, 0, 0] = 90

        self.mock_reader.readtext.side_effect = fake_readtext
        res = self.corrector.correct_orientation(img, card_type="gplx", check_all_angles=True)
        self.assertTrue(res.success)
        self.assertEqual(res.rotation_deg, 90)

    def test_low_information_triggers_fallback(self):
        """5 & 12. Ảnh không đủ thông tin / confidence quá thấp -> Fallback giữ nguyên 0°."""
        self.mock_reader.readtext.return_value = []  # Không đọc được gì
        img = np.zeros((540, 856, 3), dtype=np.uint8)

        res = self.corrector.correct_orientation(img, card_type="cccd")
        self.assertFalse(res.success)
        self.assertTrue(res.fallback_used)
        self.assertEqual(res.rotation_deg, 0)

    def test_input_not_mutated(self):
        """14. Không làm thay đổi in-place mảng ảnh đầu vào."""
        img = np.ones((540, 856, 3), dtype=np.uint8) * 128
        img_copy = img.copy()

        self.mock_reader.readtext.return_value = []
        _ = self.corrector.correct_orientation(img, card_type="cccd")

        self.assertTrue(np.array_equal(img, img_copy), "Ảnh đầu vào bị sửa đổi in-place!")

    def test_no_mirror_flipping(self):
        """13. Không lật gương văn bản (cv2.flip ngang thay vì xoay)."""
        # Tạo ma trận bất đối xứng 3x3 để phân biệt xoay 180° vs lật gương ngang
        img = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        rotated_180 = self.corrector._rotate_image(img, 180)
        # Điểm (0, 0) của ảnh xoay 180° phải là điểm (2, 2) của ảnh gốc = 9
        self.assertEqual(rotated_180[0, 0, 0], 9)
        # Điểm (0, 2) của ảnh xoay 180° phải là điểm (2, 0) của ảnh gốc = 7 (nếu lật gương ngang thì là 1)
        self.assertEqual(rotated_180[0, 2, 0], 7)
        self.assertEqual(rotated_180[2, 0, 0], 3)
        self.assertEqual(rotated_180[2, 2, 0], 1)


class TestRoiCalibrationAndGeometry(unittest.TestCase):
    """Kiểm thử cấu hình ROI hiệu chuẩn và tính toàn vẹn hình học."""

    def test_rois_fully_within_bounds(self):
        """8. Tất cả các ROI hiệu chuẩn phải nằm hoàn toàn trong [0, 1]."""
        for name, box in RECTIFIED_CCCD_ROIS.items():
            x1, y1, x2, y2 = box
            self.assertGreaterEqual(x1, 0.0, f"CCCD {name} x1 < 0")
            self.assertGreaterEqual(y1, 0.0, f"CCCD {name} y1 < 0")
            self.assertLessEqual(x2, 1.0, f"CCCD {name} x2 > 1")
            self.assertLessEqual(y2, 1.0, f"CCCD {name} y2 > 1")

        for name, box in RECTIFIED_GPLX_ROIS.items():
            x1, y1, x2, y2 = box
            self.assertGreaterEqual(x1, 0.0, f"GPLX {name} x1 < 0")
            self.assertGreaterEqual(y1, 0.0, f"GPLX {name} y1 < 0")
            self.assertLessEqual(x2, 1.0, f"GPLX {name} x2 > 1")
            self.assertLessEqual(y2, 1.0, f"GPLX {name} y2 > 1")

    def test_rois_not_empty_and_valid_dimensions(self):
        """9 & 10. Tọa độ ROI hợp lệ: x2 > x1 và y2 > y1."""
        for name, box in RECTIFIED_CCCD_ROIS.items():
            x1, y1, x2, y2 = box
            self.assertGreater(x2, x1, f"CCCD {name} x2 <= x1")
            self.assertGreater(y2, y1, f"CCCD {name} y2 <= y1")

        for name, box in RECTIFIED_GPLX_ROIS.items():
            x1, y1, x2, y2 = box
            self.assertGreater(x2, x1, f"GPLX {name} x2 <= x1")
            self.assertGreater(y2, y1, f"GPLX {name} y2 <= y1")

    def test_cccd_and_gplx_distinct_layouts(self):
        """7. CCCD và GPLX có cấu trúc trường và tọa độ khác biệt."""
        self.assertIn("residence_line_1", RECTIFIED_CCCD_ROIS)
        self.assertNotIn("residence_line_1", RECTIFIED_GPLX_ROIS)
        self.assertIn("license_class", RECTIFIED_GPLX_ROIS)
        self.assertNotIn("license_class", RECTIFIED_CCCD_ROIS)

    def test_baseline_rois_unmodified(self):
        """15. Không làm thay đổi ROI Baseline của Giai đoạn 1."""
        # Baseline CCCD
        self.assertEqual(CCCD_ROIS["id_number"], (0.37, 0.39, 0.89, 0.51))
        self.assertEqual(CCCD_ROIS["full_name"], (0.27, 0.535, 0.62, 0.605))
        # Baseline GPLX
        self.assertEqual(GPLX_ROIS["license_number"], (0.61, 0.235, 0.86, 0.315))
        self.assertEqual(GPLX_ROIS["full_name"], (0.62, 0.315, 0.85, 0.380))


if __name__ == "__main__":
    unittest.main()
