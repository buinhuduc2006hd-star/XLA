"""Bộ Unit Test cho module DocumentDetector (Giai đoạn 2 - YOLOv8n Document Detection).

Chạy kiểm thử:
    python -m unittest tests.test_document_detector -v

Bao phủ các ca kiểm thử bắt buộc:
1. Không phát hiện được thẻ (Zero detection / No card in image).
2. Phát hiện 1 thẻ hợp lệ (Single document detection).
3. Phát hiện nhiều thẻ trong cùng một ảnh (Multiple documents detection & sorting).
4. Xử lý ảnh không hợp lệ (Invalid inputs: corrupt, wrong dims, non-existent, empty array).
5. Trực quan hóa và lưu bounding box (annotate_image).
6. Toàn vẹn cấu hình data.yaml và bộ kiểm tra dataset (dataset_prepare).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch

# Thêm thư mục cccd-reader-api-main vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.document_detector import (
    CLASS_ID_CCCD,
    CLASS_ID_GPLX,
    CLASS_NAMES,
    Detection,
    DetectionResult,
    DocumentDetector,
)


class TestDocumentDetectorInputValidation(unittest.TestCase):
    """Kiểm thử các trường hợp đầu vào không hợp lệ."""

    def setUp(self):
        # Mock YOLO model để không cần tải trọng số nặng khi test validation
        with patch("services.document_detector.YOLO"):
            self.detector = DocumentDetector(model_path="dummy.pt")

    def test_non_existent_file_raises_filenotfound(self):
        """File ảnh không tồn tại -> Bắt buộc raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.detector.detect("non_existent_image_123456.jpg")

    def test_unsupported_type_raises_typeerror(self):
        """Đầu vào sai kiểu dữ liệu (số nguyên, list...) -> Bắt buộc raise TypeError."""
        with self.assertRaises(TypeError):
            self.detector.detect(12345)  # type: ignore

        with self.assertRaises(TypeError):
            self.detector.detect(["invalid", "path"])  # type: ignore

    def test_empty_numpy_array_raises_valueerror(self):
        """Mảng NumPy rỗng -> Raise ValueError."""
        empty_arr = np.array([], dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.detector.detect(empty_arr)

    def test_wrong_dimensions_raises_valueerror(self):
        """Mảng 1D, 2D (grayscale) hoặc 4D -> Raise ValueError."""
        # 1D array
        with self.assertRaises(ValueError):
            self.detector.detect(np.zeros((100,), dtype=np.uint8))

        # 2D array (grayscale không có channel)
        with self.assertRaises(ValueError):
            self.detector.detect(np.zeros((100, 100), dtype=np.uint8))

        # 4-channel RGBA
        with self.assertRaises(ValueError):
            self.detector.detect(np.zeros((100, 100, 4), dtype=np.uint8))

    def test_too_small_dimensions_raises_valueerror(self):
        """Ảnh có kích thước quá nhỏ (< 10x10) -> Raise ValueError."""
        tiny_img = np.zeros((8, 8, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.detector.detect(tiny_img)

    def test_corrupt_file_raises_valueerror(self):
        """File không phải định dạng ảnh hoặc bị hỏng -> Raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"NOT_A_REAL_IMAGE_BYTES")
            temp_path = f.name

        try:
            with self.assertRaises(ValueError):
                self.detector.detect(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestDocumentDetectorInferenceCases(unittest.TestCase):
    """Kiểm thử logic inference: không có thẻ, 1 thẻ, nhiều thẻ."""

    def _create_mock_boxes(self, boxes_data: list[tuple[int, float, list[float]]]):
        """Tạo mock object tương thích với Ultralytics Results.boxes.

        boxes_data: list of (class_id, conf, [x1, y1, x2, y2])
        """
        if not boxes_data:
            mock_boxes = MagicMock()
            mock_boxes.__len__.return_value = 0
            mock_boxes.xyxy = torch.zeros((0, 4))
            mock_boxes.conf = torch.zeros((0,))
            mock_boxes.cls = torch.zeros((0,))
            return mock_boxes

        clss = [b[0] for b in boxes_data]
        confs = [b[1] for b in boxes_data]
        xyxy = [b[2] for b in boxes_data]

        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = len(boxes_data)
        mock_boxes.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        mock_boxes.conf = torch.tensor(confs, dtype=torch.float32)
        mock_boxes.cls = torch.tensor(clss, dtype=torch.float32)
        return mock_boxes

    def setUp(self):
        self.mock_yolo = MagicMock()
        self.mock_yolo.names = {0: "cccd", 1: "gplx"}
        self.dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    @patch("services.document_detector.YOLO")
    def test_case_no_detection(self, mock_yolo_cls):
        """Trường hợp 1: Không phát hiện được thẻ nào (No card in image)."""
        mock_yolo_cls.return_value = self.mock_yolo

        # Mock YOLO predict trả về rỗng
        mock_res = MagicMock()
        mock_res.boxes = self._create_mock_boxes([])
        self.mock_yolo.predict.return_value = [mock_res]

        detector = DocumentDetector(model_path="dummy.pt")
        result = detector.detect(self.dummy_img)

        self.assertIsInstance(result, DetectionResult)
        self.assertFalse(result.has_detection)
        self.assertEqual(len(result.detections), 0)
        self.assertIsNone(result.top_detection)
        self.assertEqual(result.image_shape, (480, 640))

    @patch("services.document_detector.YOLO")
    def test_case_single_detection_cccd(self, mock_yolo_cls):
        """Trường hợp 2: Phát hiện 1 thẻ CCCD hợp lệ."""
        mock_yolo_cls.return_value = self.mock_yolo

        mock_res = MagicMock()
        # class=0 (cccd), conf=0.92, bbox=[50, 60, 400, 300]
        mock_res.boxes = self._create_mock_boxes([(0, 0.92, [50.0, 60.0, 400.0, 300.0])])
        self.mock_yolo.predict.return_value = [mock_res]

        detector = DocumentDetector(model_path="dummy.pt")
        result = detector.detect(self.dummy_img)

        self.assertTrue(result.has_detection)
        self.assertEqual(len(result.detections), 1)
        self.assertIsNotNone(result.top_detection)

        det = result.top_detection
        self.assertEqual(det.class_id, CLASS_ID_CCCD)
        self.assertEqual(det.class_name, "cccd")
        self.assertAlmostEqual(det.confidence, 0.92, places=2)
        self.assertEqual(det.bbox_xyxy, (50, 60, 400, 300))

    @patch("services.document_detector.YOLO")
    def test_case_single_detection_gplx(self, mock_yolo_cls):
        """Trường hợp 3: Phát hiện 1 thẻ GPLX hợp lệ."""
        mock_yolo_cls.return_value = self.mock_yolo

        mock_res = MagicMock()
        # class=1 (gplx), conf=0.88, bbox=[30, 40, 500, 350]
        mock_res.boxes = self._create_mock_boxes([(1, 0.88, [30.0, 40.0, 500.0, 350.0])])
        self.mock_yolo.predict.return_value = [mock_res]

        detector = DocumentDetector(model_path="dummy.pt")
        result = detector.detect(self.dummy_img)

        self.assertTrue(result.has_detection)
        self.assertEqual(result.top_detection.class_id, CLASS_ID_GPLX)
        self.assertEqual(result.top_detection.class_name, "gplx")
        self.assertAlmostEqual(result.top_detection.confidence, 0.88, places=2)

    @patch("services.document_detector.YOLO")
    def test_case_multiple_detections_sorting(self, mock_yolo_cls):
        """Trường hợp 4: Phát hiện nhiều thẻ (Multiple cards) -> Sắp xếp giảm dần theo confidence."""
        mock_yolo_cls.return_value = self.mock_yolo

        mock_res = MagicMock()
        # Thẻ 1: GPLX conf 0.75
        # Thẻ 2: CCCD conf 0.95 (cao hơn)
        # Thẻ 3: CCCD conf 0.60
        mock_res.boxes = self._create_mock_boxes([
            (1, 0.75, [100.0, 100.0, 300.0, 250.0]),
            (0, 0.95, [320.0, 100.0, 550.0, 250.0]),
            (0, 0.60, [150.0, 280.0, 450.0, 440.0]),
        ])
        self.mock_yolo.predict.return_value = [mock_res]

        detector = DocumentDetector(model_path="dummy.pt")
        result = detector.detect(self.dummy_img)

        self.assertTrue(result.has_detection)
        self.assertEqual(len(result.detections), 3)

        # Kiểm tra top_detection phải là phần tử có confidence cao nhất (~0.95)
        self.assertAlmostEqual(result.top_detection.confidence, 0.95, places=2)
        self.assertEqual(result.top_detection.class_name, "cccd")

        # Kiểm tra thứ tự sắp xếp giảm dần
        confs = [d.confidence for d in result.detections]
        self.assertGreater(confs[0], confs[1])
        self.assertGreater(confs[1], confs[2])

    @patch("services.document_detector.YOLO")
    def test_bbox_coordinates_clamped_to_image_boundary(self, mock_yolo_cls):
        """Tọa độ bbox ngoài rìa ảnh phải được kẹp (clamp) an toàn trong [0, W-1] và [0, H-1]."""
        mock_yolo_cls.return_value = self.mock_yolo

        mock_res = MagicMock()
        # BBox có tọa độ âm hoặc vượt quá kích thước 640x480
        mock_res.boxes = self._create_mock_boxes([(0, 0.90, [-20.0, -10.0, 700.0, 500.0])])
        self.mock_yolo.predict.return_value = [mock_res]

        detector = DocumentDetector(model_path="dummy.pt")
        result = detector.detect(self.dummy_img)

        x1, y1, x2, y2 = result.top_detection.bbox_xyxy
        self.assertEqual(x1, 0)
        self.assertEqual(y1, 0)
        self.assertEqual(x2, 639)
        self.assertEqual(y2, 479)


class TestDocumentDetectorAnnotation(unittest.TestCase):
    """Kiểm thử tính năng vẽ và lưu ảnh annotated."""

    def test_annotate_image_and_save(self):
        """Vẽ bounding box lên ảnh và lưu ra file tạm thời thành công."""
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        detections = [
            Detection(class_id=0, class_name="cccd", confidence=0.91, bbox_xyxy=(50, 50, 200, 180)),
            Detection(class_id=1, class_name="gplx", confidence=0.85, bbox_xyxy=(220, 50, 380, 180)),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "output_annotated.png"
            annotated = DocumentDetector.annotate_image(img, detections, save_path=save_path)

            self.assertEqual(annotated.shape, (300, 400, 3))
            self.assertTrue(save_path.exists())
            self.assertGreater(save_path.stat().st_size, 0)


class TestDatasetConfiguration(unittest.TestCase):
    """Kiểm tra file cấu hình data.yaml và script dataset_prepare."""

    def test_data_yaml_exists_and_valid(self):
        """File data.yaml phải tồn tại và chứa đúng 2 lớp cccd & gplx."""
        import yaml

        yaml_path = PROJECT_ROOT / "configs" / "data.yaml"
        self.assertTrue(yaml_path.exists(), f"Không tìm thấy file: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.assertEqual(config.get("nc"), 2)
        names = config.get("names")
        self.assertEqual(names[0], "cccd")
        self.assertEqual(names[1], "gplx")
        self.assertIn("train", config)
        self.assertIn("val", config)

    def test_dataset_prepare_checker(self):
        """Hàm check_dataset_integrity chạy không crash trên thư mục datasets."""
        from scripts.train.dataset_prepare import check_dataset_integrity

        dataset_root = PROJECT_ROOT / "datasets"
        report = check_dataset_integrity(dataset_root)
        self.assertIn("is_valid", report)
        self.assertIn("splits", report)
        self.assertIn("class_counts", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
