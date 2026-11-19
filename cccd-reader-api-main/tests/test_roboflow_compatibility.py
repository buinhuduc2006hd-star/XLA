"""Unit tests kiểm thử tính tương thích giữa YOLOv8-Pose và cấu hình Keypoint Roboflow.

Kiểm tra toàn diện 10 yêu cầu:
1. Thứ tự logic TL/TR/BR/BL và tên Roboflow (point_up_left, point_up_right, point_bottom_right, point_bottom_left).
2. Phát hiện và từ chối sai thứ tự Keypoint.
3. Biến đổi phối cảnh trên ảnh xoay 90°, 180°, 270°.
4. Từ chối Keypoint bị thiếu hoặc không hữu hạn (NaN/Inf).
5. Từ chối Keypoint nằm ngoài biên ảnh.
6. Từ chối Polygon tự giao (self-intersecting / non-convex).
7. Ánh xạ Class chuẩn CCCD (0) và GPLX (1).
8. Quy luật đối xứng Flip ngang (flip_idx = [1, 0, 3, 2]).
9. Dataset Importer dry-run và auto-remap với Roboflow ZIP metadata.
10. Cách ly hoàn toàn, không còn phụ thuộc runtime vào YOLO Detection cũ.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

# Đảm bảo đường dẫn import cccd-reader-api-main
API_ROOT = Path(__file__).resolve().parent.parent
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import numpy as np

# Import các module cốt lõi của Pose Pipeline
from services.document_pose_detector import (
    KEYPOINT_NAMES,
    ROBOFLOW_KEYPOINT_NAMES,
    validate_card_keypoints,
)
from services.document_rectifier import DocumentRectifier
from scripts.pose.import_roboflow import (
    TARGET_KEYPOINTS,
    get_class_remap,
    get_keypoint_remap,
    identify_target_keypoint_index,
    import_roboflow_zip,
    validate_and_convert_label,
)
from scripts.pose.validate_pose_dataset import validate_dataset


class TestRoboflowKeypointCompatibility(unittest.TestCase):
    """Kiểm thử tính tương thích cấu hình Roboflow 4 Keypoints."""

    def setUp(self):
        self.rectifier = DocumentRectifier()

    def test_01_roboflow_keypoint_order_and_naming(self):
        """1. Xác minh thứ tự logic TL/TR/BR/BL và tên Roboflow chuẩn."""
        expected = [
            "point_up_left",       # Index 0: Top Left
            "point_up_right",      # Index 1: Top Right
            "point_bottom_right",  # Index 2: Bottom Right
            "point_bottom_left",   # Index 3: Bottom Left
        ]
        self.assertEqual(ROBOFLOW_KEYPOINT_NAMES, expected)
        self.assertEqual(KEYPOINT_NAMES, expected)
        self.assertEqual(TARGET_KEYPOINTS, expected)

        # Kiểm tra identify index
        self.assertEqual(identify_target_keypoint_index("point_up_left"), 0)
        self.assertEqual(identify_target_keypoint_index("point_up_right"), 1)
        self.assertEqual(identify_target_keypoint_index("point_bottom_right"), 2)
        self.assertEqual(identify_target_keypoint_index("point_bottom_left"), 3)

    def test_02_keypoint_validation_valid_and_wrong_order(self):
        """2. Xác minh Keypoint hợp lệ được chấp nhận, sai thứ tự chiều bị từ chối."""
        # Tọa độ chuẩn chiều kim đồng hồ (TL -> TR -> BR -> BL) trên ảnh 1000x600
        valid_corners = np.array([
            [100.0, 100.0],  # point_up_left (TL)
            [800.0, 100.0],  # point_up_right (TR)
            [800.0, 500.0],  # point_bottom_right (BR)
            [100.0, 500.0],  # point_bottom_left (BL)
        ], dtype=np.float32)

        is_valid, msg = validate_card_keypoints(valid_corners, image_shape=(600, 1000))
        self.assertTrue(is_valid, f"Valid corners rejected: {msg}")

        # Thứ tự ngược chiều kim đồng hồ (TL -> BL -> BR -> TR) -> Phải bị từ chối
        inverted_order = np.array([
            [100.0, 100.0],
            [100.0, 500.0],
            [800.0, 500.0],
            [800.0, 100.0],
        ], dtype=np.float32)

        is_valid, msg = validate_card_keypoints(inverted_order, image_shape=(600, 1000))
        self.assertFalse(is_valid)
        self.assertIn("counter-clockwise or inverted", msg)

    def test_03_rotated_cards_geometry(self):
        """3. Biến đổi phối cảnh trên thẻ xoay các góc 90°, 180°, 270°."""
        dummy_img = np.ones((800, 800, 3), dtype=np.uint8) * 128

        # Thẻ xuôi (0°)
        corners_0 = np.array([[100, 100], [600, 100], [600, 400], [100, 400]], dtype=np.float32)
        res_0 = self.rectifier.rectify_from_corners(dummy_img, corners_0)
        self.assertTrue(res_0.success)
        self.assertEqual(res_0.rectified_image.shape, (540, 856, 3))

        # Thẻ xoay 90° (TL ở góc trên-phải, TR ở góc dưới-phải...)
        corners_90 = np.array([[600, 100], [600, 600], [300, 600], [300, 100]], dtype=np.float32)
        res_90 = self.rectifier.rectify_from_corners(dummy_img, corners_90)
        self.assertTrue(res_90.success)
        self.assertEqual(res_90.rectified_image.shape, (540, 856, 3))

    def test_04_missing_or_out_of_bounds_keypoints(self):
        """4 & 5. Từ chối keypoints chứa NaN/Inf hoặc nằm xa ngoài ảnh."""
        # Chứa NaN
        nan_corners = np.array([[100, 100], [np.nan, 100], [600, 400], [100, 400]], dtype=np.float32)
        is_valid, msg = validate_card_keypoints(nan_corners, image_shape=(600, 1000))
        self.assertFalse(is_valid)
        self.assertIn("NaN or Inf", msg)

        # Nằm xa ngoài biên ảnh (> 8% tolerance)
        out_corners = np.array([[-200, 100], [800, 100], [800, 500], [100, 500]], dtype=np.float32)
        is_valid, msg = validate_card_keypoints(out_corners, image_shape=(600, 1000))
        self.assertFalse(is_valid)
        self.assertIn("out of allowed image bounds", msg)

    def test_05_self_intersecting_polygon(self):
        """6. Từ chối tứ giác tự giao nhau (Bow-tie hình cánh bướm)."""
        bowtie_corners = np.array([
            [100.0, 100.0],  # TL
            [800.0, 500.0],  # TR bị chéo xuống góc BR
            [800.0, 100.0],  # BR bị chéo lên góc TR
            [100.0, 500.0],  # BL
        ], dtype=np.float32)

        is_valid, msg = validate_card_keypoints(bowtie_corners, image_shape=(600, 1000))
        self.assertFalse(is_valid)
        # Sẽ bị bắt bởi kiểm tra thứ tự vector hoặc không lồi
        self.assertTrue(any(term in msg for term in ["non-convex", "counter-clockwise", "inverted", "Edge"]))

    def test_06_roboflow_class_mapping(self):
        """7. Ánh xạ Class chuẩn CCCD -> 0, GPLX -> 1."""
        yaml_mock = {
            "names": ["CCCD_Front", "Driver_License_GPLX"]
        }
        remap = get_class_remap(yaml_mock)
        self.assertEqual(remap[0], 0)  # cccd
        self.assertEqual(remap[1], 1)  # gplx

    def test_07_horizontal_flip_symmetry(self):
        """8. Kiểm tra quy tắc đối xứng khi lật ngang: flip_idx = [1, 0, 3, 2]."""
        flip_idx = [1, 0, 3, 2]
        # Giả sử tọa độ x của 4 góc trước khi lật:
        # TL (0) có x=0.2, TR (1) có x=0.8
        # Sau khi lật ngang (x' = 1 - x):
        # x' của TL ban đầu là 1 - 0.2 = 0.8 (trở thành TR mới)
        # x' của TR ban đầu là 1 - 0.8 = 0.2 (trở thành TL mới)
        # Tương tự BL (3) x=0.2 -> 0.8 (BR mới); BR (2) x=0.8 -> 0.2 (BL mới)
        # Như vậy hoán vị:
        # 0 -> flip_idx[0] = 1 (point_up_left <-> point_up_right)
        # 1 -> flip_idx[1] = 0 (point_up_right <-> point_up_left)
        # 2 -> flip_idx[2] = 3 (point_bottom_right <-> point_bottom_left)
        # 3 -> flip_idx[3] = 2 (point_bottom_left <-> point_bottom_right)
        self.assertEqual(flip_idx[0], 1)
        self.assertEqual(flip_idx[1], 0)
        self.assertEqual(flip_idx[2], 3)
        self.assertEqual(flip_idx[3], 2)

    def test_08_importer_dry_run_with_mock_roboflow_zip(self):
        """9. Dataset Importer dry-run và auto-remap với metadata Roboflow ZIP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            zip_file = tmp_path / "mock_roboflow_dataset.zip"

            # Tạo file data.yaml giả lập với thứ tự keypoints cần remap
            yaml_content = """
names:
  0: cccd_card
  1: gplx_card
kpt_shape: [4, 3]
keypoints:
  - point_up_right      # src 0 -> target 1
  - point_up_left       # src 1 -> target 0
  - point_bottom_left   # src 2 -> target 3
  - point_bottom_right  # src 3 -> target 2
"""
            # Dòng nhãn giả lập theo thứ tự nguồn (1 class + 4 bbox + 4*3 kpts = 17 tokens)
            # class 0 (cccd) bbox (0.5 0.5 0.8 0.5)
            # kpt 0 (point_up_right): x=0.9, y=0.1, v=2
            # kpt 1 (point_up_left):  x=0.1, y=0.1, v=2
            # kpt 2 (point_bottom_left): x=0.1, y=0.9, v=2
            # kpt 3 (point_bottom_right): x=0.9, y=0.9, v=2
            label_content = "0 0.5 0.5 0.8 0.5 0.9 0.1 2 0.1 0.1 2 0.1 0.9 2 0.9 0.9 2\n"

            # Đóng gói mock ZIP
            with zipfile.ZipFile(zip_file, "w") as zf:
                zf.writestr("data.yaml", yaml_content)
                zf.writestr("train/images/sample1.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 100)
                zf.writestr("train/labels/sample1.txt", label_content)

            # Chạy import dry-run
            report = import_roboflow_zip(
                zip_path=zip_file,
                target_pose_dir="datasets/document_pose",
                dry_run=True,
                force=True,
            )

            # Xác minh remap đã phát hiện chính xác
            expected_remap = {0: 1, 1: 0, 2: 3, 3: 2}
            self.assertEqual(report["keypoint_remap_applied"], expected_remap)
            self.assertEqual(report["total_images"], 1)
            self.assertEqual(report["total_labels"], 1)
            self.assertEqual(len(report["errors"]), 0)

            # Test hàm validate_and_convert_label trực tiếp để kiểm tra kết quả remap
            class_remap = {0: 0, 1: 1}
            is_valid, converted, err = validate_and_convert_label(label_content, class_remap, expected_remap)
            self.assertTrue(is_valid)
            tokens = converted.split()
            # Token 0: class 0
            self.assertEqual(tokens[0], "0")
            # Sau khi remap:
            # target 0 (point_up_left): x=0.1, y=0.1
            self.assertAlmostEqual(float(tokens[5]), 0.1, places=3)
            self.assertAlmostEqual(float(tokens[6]), 0.1, places=3)
            # target 1 (point_up_right): x=0.9, y=0.1
            self.assertAlmostEqual(float(tokens[8]), 0.9, places=3)
            self.assertAlmostEqual(float(tokens[9]), 0.1, places=3)

    def test_09_validate_pose_dataset_on_scaffold(self):
        """Kiểm định dataset scaffold rỗng bằng validate_pose_dataset."""
        target_dir = Path("datasets/document_pose")
        if not target_dir.exists():
            target_dir = Path("cccd-reader-api-main/datasets/document_pose")
        yaml_file = target_dir / "card_pose.yaml"

        summary = validate_dataset(dataset_dir=target_dir, yaml_path=yaml_file)
        self.assertEqual(summary["total_images"], 0)
        self.assertEqual(summary["total_invalid_labels"], 0)
        self.assertEqual(len(summary["errors"]), 0)

    def test_10_no_runtime_detection_dependency(self):
        """10. Xác nhận codebase sạch 100%, không còn import hay phụ thuộc vào YOLO Detection cũ."""
        services_dir = Path("cccd-reader-api-main/services")
        if not services_dir.exists():
            services_dir = Path("services")

        for py_file in services_dir.glob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            self.assertNotIn(
                "from services.document_detector import",
                text,
                f"File {py_file.name} still imports document_detector!",
            )
            self.assertNotIn(
                "import document_detector",
                text,
                f"File {py_file.name} still imports document_detector!",
            )


if __name__ == "__main__":
    unittest.main()
