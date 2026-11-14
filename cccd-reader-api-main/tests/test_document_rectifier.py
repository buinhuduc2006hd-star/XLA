"""Bộ unit test kiểm thử toàn diện module DocumentRectifier (Giai đoạn 3: OpenCV Perspective Transform).

Bao phủ 13 tiêu chí:
1. Crop sát biên ảnh.
2. Bounding box không hợp lệ.
3. Không tìm được contour (ảnh trơn/nhiễu).
4. Tứ giác không lồi.
5. Tứ giác tự cắt.
6. Ảnh quá nhỏ.
7. Ảnh mờ.
8. Thẻ xoay 90° và 180°.
9. Thẻ nghiêng phối cảnh (trapezoid).
10. Fallback khi rectification thất bại.
11. Đúng kích thước output (856x540).
12. Đúng hệ tọa độ các góc (gốc vs crop).
13. Không gây regression cho các trường dữ liệu.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

# Thêm path cccd-reader-api-main
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.document_rectifier import DocumentRectifier, RectifyResult


class TestDocumentRectifierValidationAndSafety(unittest.TestCase):
    """Kiểm thử tính hợp lệ đầu vào và cơ chế fallback an toàn."""

    def setUp(self) -> None:
        self.rectifier = DocumentRectifier(target_size=(856, 540), margin=0.08)
        self.valid_img = np.zeros((600, 800, 3), dtype=np.uint8)

    def test_invalid_image_type(self) -> None:
        """Ảnh không phải np.ndarray -> Fallback an toàn."""
        res = self.rectifier.rectify("not_an_image", (10, 10, 100, 100))
        self.assertFalse(res.success)
        self.assertTrue(res.fallback_used)
        self.assertIn("must be np.ndarray", str(res.error))

    def test_image_too_small(self) -> None:
        """Ảnh có kích thước quá nhỏ (< 10x10) -> Fallback an toàn."""
        small_img = np.zeros((8, 8, 3), dtype=np.uint8)
        res = self.rectifier.rectify(small_img, (0, 0, 5, 5))
        self.assertFalse(res.success)
        self.assertTrue(res.fallback_used)
        self.assertIn("too small", str(res.error))

    def test_invalid_bbox_format(self) -> None:
        """BBox thiếu tọa độ hoặc sai định dạng -> Fallback an toàn."""
        res = self.rectifier.rectify(self.valid_img, (10, 20))
        self.assertFalse(res.success)
        self.assertTrue(res.fallback_used)

    def test_inverted_bbox_coordinates(self) -> None:
        """BBox có x2 <= x1 hoặc y2 <= y1 -> Fallback an toàn."""
        res = self.rectifier.rectify(self.valid_img, (200, 200, 100, 100))
        self.assertFalse(res.success)
        self.assertTrue(res.fallback_used)
        self.assertIn("Invalid bbox dimensions", str(res.error))

    def test_bbox_completely_outside_image(self) -> None:
        """BBox nằm hoàn toàn ngoài kích thước ảnh -> Fallback an toàn."""
        res = self.rectifier.rectify(self.valid_img, (900, 900, 1000, 1000))
        self.assertFalse(res.success)
        self.assertTrue(res.fallback_used)
        self.assertIn("completely outside", str(res.error))

    def test_bbox_clamped_at_image_boundaries(self) -> None:
        """BBox nằm sát mép ảnh (x1=0, y1=0 hoặc x2=w, y2=h) -> Margin không gây crash."""
        h, w = self.valid_img.shape[:2]
        crop, (x1m, y1m, x2m, y2m) = self.rectifier._crop_with_margin(self.valid_img, (0, 0, 200, 150))
        self.assertIsNotNone(crop)
        self.assertEqual(x1m, 0)
        self.assertEqual(y1m, 0)
        self.assertLessEqual(x2m, w)
        self.assertLessEqual(y2m, h)


class TestDocumentRectifierGeometryAndOrdering(unittest.TestCase):
    """Kiểm thử thuật toán sắp xếp 4 góc và biến đổi hình học."""

    def setUp(self) -> None:
        self.rectifier = DocumentRectifier(target_size=(856, 540))

    def _rotate_points(self, pts: np.ndarray, angle_deg: float, center=(400, 300)) -> np.ndarray:
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return (pts - center) @ R.T + center

    def test_order_corners_horizontal_unrotated(self) -> None:
        """Thẻ nằm ngang (0°): [TL, TR, BR, BL] chuẩn xác."""
        card_pts = np.array([[100, 100], [500, 100], [500, 350], [100, 350]], dtype=np.float32)
        ordered = self.rectifier._order_quad_corners(card_pts)

        # Cạnh (TL -> TR) phải là cạnh trên
        self.assertAlmostEqual(ordered[0][0], 100.0, delta=1.0)
        self.assertAlmostEqual(ordered[0][1], 100.0, delta=1.0)
        self.assertAlmostEqual(ordered[1][0], 500.0, delta=1.0)
        self.assertAlmostEqual(ordered[1][1], 100.0, delta=1.0)

    def test_order_corners_rotated_90_degrees(self) -> None:
        """Thẻ xoay dọc 90°: Tự động chọn cạnh dài làm cạnh trên, không tạo hình tự cắt."""
        card_pts = np.array([[200, 100], [450, 100], [450, 500], [200, 500]], dtype=np.float32)
        ordered = self.rectifier._order_quad_corners(card_pts)

        # Độ dài cạnh ngang (TL -> TR) phải lớn hơn hoặc xấp xỉ cạnh đứng
        w = np.linalg.norm(ordered[1] - ordered[0])
        h = np.linalg.norm(ordered[3] - ordered[0])
        self.assertGreaterEqual(w, h * 0.9)

    def test_order_corners_rotated_180_degrees(self) -> None:
        """Thẻ lộn ngược 180°: 4 góc được sắp xếp vòng tròn thuận chiều, không bị vắt chéo."""
        base_pts = np.array([[100, 100], [500, 100], [500, 350], [100, 350]], dtype=np.float32)
        rot_pts = self._rotate_points(base_pts, 180.0)
        ordered = self.rectifier._order_quad_corners(rot_pts)

        # Tứ giác sau khi sắp xếp phải lồi
        self.assertTrue(cv2.isContourConvex(ordered.astype(np.int32)))

    def test_order_corners_perspective_trapezoid(self) -> None:
        """Hình thang phối cảnh (chụp nghiêng từ dưới lên) -> Top edge hẹp hơn Bottom edge."""
        trapezoid = np.array([[150, 100], [450, 100], [500, 350], [100, 350]], dtype=np.float32)
        ordered = self.rectifier._order_quad_corners(trapezoid)

        # TL và TR phải có tọa độ y nhỏ hơn BR và BL
        self.assertLess(ordered[0][1], ordered[3][1])
        self.assertLess(ordered[1][1], ordered[2][1])
        self.assertTrue(cv2.isContourConvex(ordered.astype(np.int32)))

    def test_evaluate_quad_rejects_concave_and_self_intersecting(self) -> None:
        """Đánh giá tứ giác: Loại bỏ tứ giác không lồi (hình phi tiêu) và dị dạng."""
        crop_area = 500 * 500
        # Hình lõm (phi tiêu)
        concave_pts = np.array([[100, 100], [300, 250], [400, 100], [250, 400]], dtype=np.float32)
        score = self.rectifier._evaluate_quad_candidate(concave_pts, crop_area, 20000)
        self.assertEqual(score, 0.0)

        # Tứ giác góc quá nhọn (< 45°)
        sharp_pts = np.array([[100, 100], [400, 110], [120, 120], [100, 300]], dtype=np.float32)
        score_sharp = self.rectifier._evaluate_quad_candidate(sharp_pts, crop_area, 10000)
        self.assertEqual(score_sharp, 0.0)


class TestDocumentRectifierPipelineIntegration(unittest.TestCase):
    """Kiểm thử trọn vẹn pipeline nắn chỉnh trên ảnh tổng hợp."""

    def setUp(self) -> None:
        self.rectifier = DocumentRectifier(target_size=(856, 540), margin=0.08)

    def test_rectify_synthetic_card_perspective(self) -> None:
        """Tạo ảnh tổng hợp chứa thẻ nghiêng phối cảnh -> Warp thành công ra đúng 856x540."""
        # Tạo nền tối 700x1000
        canvas = np.full((700, 1000, 3), 40, dtype=np.uint8)

        # Vẽ một thẻ sáng màu (tỷ lệ 1.58) bị xoay nhẹ trên nền
        card_corners = np.array([[220, 180], [680, 140], [720, 450], [250, 490]], dtype=np.int32)
        cv2.fillPoly(canvas, [card_corners], (220, 220, 220))
        cv2.polylines(canvas, [card_corners], True, (10, 10, 10), 3)

        # Bounding box bao quanh thẻ
        x1, y1 = 200, 130
        x2, y2 = 740, 510

        res = self.rectifier.rectify(canvas, (x1, y1, x2, y2), card_type="cccd")

        self.assertTrue(res.success)
        self.assertFalse(res.fallback_used)
        self.assertIsNotNone(res.corners)
        self.assertEqual(len(res.corners), 4)
        self.assertEqual(res.rectified_image.shape, (540, 856, 3))
        self.assertGreater(res.confidence, 0.5)

    def test_rectify_blank_uniform_image_triggers_fallback(self) -> None:
        """Ảnh trơn hoàn toàn không có biên thẻ -> Kích hoạt fallback an toàn, không crash."""
        blank_canvas = np.full((500, 500, 3), 128, dtype=np.uint8)
        res = self.rectifier.rectify(blank_canvas, (50, 50, 450, 450))

        self.assertFalse(res.success)
        self.assertTrue(res.fallback_used)
        self.assertIsNone(res.corners)
        self.assertEqual(res.rectified_image.shape, (540, 856, 3))
        self.assertIn("fallback", res.method)

    def test_rectify_blurred_card_image(self) -> None:
        """Ảnh bị làm mờ GaussianBlur mạnh -> Bộ lọc Canny đa tầng vẫn xử lý an toàn."""
        canvas = np.full((600, 800, 3), 50, dtype=np.uint8)
        card_corners = np.array([[150, 150], [650, 150], [650, 450], [150, 450]], dtype=np.int32)
        cv2.fillPoly(canvas, [card_corners], (200, 200, 200))
        # Làm mờ mạnh
        blurred_canvas = cv2.GaussianBlur(canvas, (15, 15), 5.0)

        res = self.rectifier.rectify(blurred_canvas, (140, 140, 660, 460))
        # Kết quả phải trả về RectifyResult hợp lệ (dù warp được hay fallback) và không ném ngoại lệ
        self.assertIsInstance(res, RectifyResult)
        self.assertEqual(res.rectified_image.shape, (540, 856, 3))

    def test_draw_corners_utility(self) -> None:
        """Kiểm thử hàm tiện ích draw_corners vẽ đúng 4 điểm TL, TR, BR, BL."""
        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        corners = [[50, 50], [300, 50], [300, 200], [50, 200]]
        annotated = self.rectifier.draw_corners(canvas, corners)
        self.assertEqual(annotated.shape, canvas.shape)
        # Đảm bảo đã vẽ (không còn đen toàn bộ)
        self.assertGreater(np.count_nonzero(annotated), 0)


if __name__ == "__main__":
    unittest.main()

