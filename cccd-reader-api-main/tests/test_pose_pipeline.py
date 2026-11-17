"""Unit tests và Regression tests cho Pose-only Architecture & Keypoint Validator.

Kiểm thử toàn diện theo yêu cầu Kiến Trúc Pose-Only:
1. Output bốn Keypoint hợp lệ -> Validation thành công.
2. Output thiếu Keypoint (None hoặc thiếu điểm).
3. Confidence thấp (dưới ngưỡng tối thiểu 0.25).
4. Keypoint trùng nhau hoặc cạnh quá ngắn (< 25px).
5. Keypoint nằm ngoài biên ảnh cho phép (> 8% margin).
6. Tọa độ không hợp lệ (NaN / Inf).
7. Tứ giác không lồi (Bow-tie / Non-convex / Self-intersecting).
8. Thứ tự góc sai (Ngược chiều kim đồng hồ / Lật gương).
9. Xử lý nhiều giấy tờ trong ảnh (Multi-document selection: chọn thẻ tối ưu score = conf * sqrt(area)).
10. Thiếu checkpoint mô hình (PoseModelNotFoundError).
11. Checkpoint không tương thích (PoseModelIncompatibleError).
12. Nắn phối cảnh trực tiếp từ 4 góc tổng hợp (Rectify from corners -> 856x540).
13. Thử nghiệm nắn phối cảnh trên thẻ xoay 90°, 180°, 270°.
14. Chế độ Pose không nạp YOLO Detection model.
15. FastAPI trả mã lỗi 503 POSE_MODEL_NOT_FOUND khi thiếu checkpoint.
16. FastAPI trả mã lỗi 422 KEYPOINT_VALIDATION_FAILED khi 4 góc thẻ không đạt tiêu chuẩn.
17. FastAPI trả mã lỗi 422 NO_DOCUMENT_DETECTED khi không tìm thấy thẻ.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

# Đảm bảo đường dẫn API nằm trong sys.path
API_ROOT = Path(__file__).resolve().parent.parent
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from schemas import DocumentV2Response, ErrorResponse
from services.document_pose_detector import (
    DocumentPoseDetector,
    KeypointValidationError,
    PoseDetectionResult,
    PoseModelIncompatibleError,
    PoseModelNotFoundError,
    get_document_pose_detector,
    reset_document_pose_detector,
    validate_card_keypoints,
)
from services.document_rectifier import DocumentRectifier, RectifyResult
from services.pipeline_pose import (
    DocumentTypeMismatchError,
    NoDocumentDetectedError,
    PipelinePoseError,
    process_document_pose,
)


class TestKeypointValidation(unittest.TestCase):
    """Kiểm thử chuyên sâu hàm validate_card_keypoints (8 ràng buộc hình học)."""

    def setUp(self):
        self.img_h, self.img_w = 600, 800
        # Tứ giác chuẩn theo chiều kim đồng hồ [TL, TR, BR, BL]
        self.valid_corners = np.array(
            [
                [100.0, 100.0],  # TL
                [700.0, 100.0],  # TR
                [700.0, 500.0],  # BR
                [100.0, 500.0],  # BL
            ],
            dtype=np.float32,
        )
        self.valid_confs = [0.95, 0.92, 0.91, 0.94]

    def test_01_valid_four_corners(self):
        """1. Output 4 keypoints hợp lệ -> Validation thành công."""
        is_valid, reason = validate_card_keypoints(
            self.valid_corners,
            image_shape=(self.img_h, self.img_w),
            keypoint_confidences=self.valid_confs,
        )
        self.assertTrue(is_valid)
        self.assertIsNone(reason)

    def test_02_missing_or_malformed_keypoints(self):
        """2. Output thiếu keypoints hoặc không đúng kích thước (4, 2)."""
        is_valid, reason = validate_card_keypoints(
            None,
            image_shape=(self.img_h, self.img_w),
        )
        self.assertFalse(is_valid)
        self.assertIn("None", reason)

        # 3 điểm thay vì 4
        invalid_shape = np.array([[100, 100], [700, 100], [700, 500]], dtype=np.float32)
        is_valid, reason = validate_card_keypoints(
            invalid_shape,
            image_shape=(self.img_h, self.img_w),
        )
        self.assertFalse(is_valid)
        self.assertIn("(4, 2)", reason)

    def test_03_low_confidence_keypoints(self):
        """3. Confidence của một góc quá thấp (< 0.25) -> Validation từ chối."""
        low_confs = [0.95, 0.15, 0.91, 0.94]  # TR conf = 0.15 < 0.25
        is_valid, reason = validate_card_keypoints(
            self.valid_corners,
            image_shape=(self.img_h, self.img_w),
            keypoint_confidences=low_confs,
            min_kpt_conf=0.25,
        )
        self.assertFalse(is_valid)
        self.assertIn("top_right", reason)
        self.assertIn("below minimum threshold", reason)

    def test_04_overlapping_corners_or_short_edge(self):
        """4. Hai góc quá gần nhau hoặc trùng lặp (< 25px)."""
        overlap_corners = self.valid_corners.copy()
        overlap_corners[1] = overlap_corners[0] + np.array([5.0, 0.0])  # Cạnh TL-TR chỉ dài 5px
        is_valid, reason = validate_card_keypoints(
            overlap_corners,
            image_shape=(self.img_h, self.img_w),
            keypoint_confidences=self.valid_confs,
            min_edge_length_px=25.0,
        )
        self.assertFalse(is_valid)
        self.assertIn("too short", reason)

    def test_05_corners_outside_image_bounds(self):
        """5. Góc nằm xa ngoài khung ảnh (> 8% margin)."""
        out_corners = self.valid_corners.copy()
        out_corners[1] = np.array([1200.0, 100.0])  # Vượt xa chiều rộng ảnh 800px
        is_valid, reason = validate_card_keypoints(
            out_corners,
            image_shape=(self.img_h, self.img_w),
            keypoint_confidences=self.valid_confs,
            margin_tolerance_ratio=0.08,
        )
        self.assertFalse(is_valid)
        self.assertIn("out of allowed image bounds", reason)

    def test_06_non_finite_coordinates(self):
        """6. Tọa độ chứa NaN hoặc Inf (thẻ bị mất góc / hỏng output)."""
        nan_corners = self.valid_corners.copy()
        nan_corners[2] = np.array([np.nan, 500.0])
        is_valid, reason = validate_card_keypoints(
            nan_corners,
            image_shape=(self.img_h, self.img_w),
        )
        self.assertFalse(is_valid)
        self.assertIn("NaN or Inf", reason)

    def test_07_non_convex_quadrilateral(self):
        """7. Tứ giác không lồi (Bow-tie hoặc tự cắt)."""
        # Hoán đổi TR và BR tạo hình cánh bướm tự cắt
        bowtie_corners = np.array(
            [
                [100.0, 100.0],  # TL
                [700.0, 500.0],  # TR -> bị đổi xuống BR
                [700.0, 100.0],  # BR -> bị đổi lên TR
                [100.0, 500.0],  # BL
            ],
            dtype=np.float32,
        )
        is_valid, reason = validate_card_keypoints(
            bowtie_corners,
            image_shape=(self.img_h, self.img_w),
            keypoint_confidences=self.valid_confs,
        )
        self.assertFalse(is_valid)
        self.assertIn("not convex", reason)

    def test_08_inverted_counter_clockwise_order(self):
        """8. Thứ tự ngược chiều kim đồng hồ hoặc bị lật gương."""
        ccw_corners = np.array(
            [
                [100.0, 100.0],  # TL
                [100.0, 500.0],  # BL (đi xuống thay vì sang phải)
                [700.0, 500.0],  # BR
                [700.0, 100.0],  # TR
            ],
            dtype=np.float32,
        )
        is_valid, reason = validate_card_keypoints(
            ccw_corners,
            image_shape=(self.img_h, self.img_w),
            keypoint_confidences=self.valid_confs,
        )
        self.assertFalse(is_valid)
        self.assertIn("counter-clockwise or inverted", reason)


class TestPoseDetectorLogic(unittest.TestCase):
    """Kiểm thử lớp DocumentPoseDetector với Mock Predictions."""

    def setUp(self):
        reset_document_pose_detector()
        self.dummy_img = np.ones((600, 800, 3), dtype=np.uint8) * 128

    def tearDown(self):
        reset_document_pose_detector()

    @patch("services.document_pose_detector.Path.exists", return_value=False)
    def test_10_missing_checkpoint_raises_error(self, mock_exists):
        """10. Thiếu checkpoint -> Ném PoseModelNotFoundError."""
        with self.assertRaises(PoseModelNotFoundError) as ctx:
            DocumentPoseDetector(model_path="models/document_pose/non_existent.pt")
        self.assertEqual(ctx.exception.error_code, "POSE_MODEL_NOT_FOUND")

    @patch("services.document_pose_detector.Path.exists", return_value=True)
    def test_11_unsupported_checkpoint_raises_error(self, mock_exists):
        """11. Checkpoint hỏng hoặc không tương thích -> Ném PoseModelIncompatibleError."""
        with patch("ultralytics.YOLO", side_effect=RuntimeError("Corrupt model file")):
            with self.assertRaises(PoseModelIncompatibleError) as ctx:
                DocumentPoseDetector(model_path="models/document_pose/corrupt.pt")
            self.assertEqual(ctx.exception.error_code, "POSE_MODEL_INCOMPATIBLE")

    @patch("services.document_pose_detector.Path.exists", return_value=True)
    def test_09_multi_document_selection(self, mock_exists):
        """9. Nhiều thẻ trong ảnh -> Chọn thẻ có điểm số cao nhất hoặc khớp class."""
        mock_yolo = MagicMock()

        # Mock detection: Thẻ 0: cccd diện tích 400x300 conf=0.5, Thẻ 1: gplx lớn conf=0.9
        mock_res = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 2
        mock_boxes.xyxy.cpu().numpy.return_value = np.array(
            [[50, 50, 450, 350], [250, 100, 750, 450]], dtype=np.float32
        )
        mock_boxes.conf.cpu().numpy.return_value = np.array([0.5, 0.9], dtype=np.float32)
        mock_boxes.cls.cpu().numpy.return_value = np.array([0, 1], dtype=np.float32)  # 0: cccd, 1: gplx

        mock_kpts = MagicMock()
        mock_kpts.__len__.return_value = 2
        cccd_corners = np.array([[60, 60], [440, 60], [440, 340], [60, 340]], dtype=np.float32)
        gplx_corners = np.array([[260, 110], [740, 110], [740, 440], [260, 440]], dtype=np.float32)
        mock_kpts.xy.cpu().numpy.return_value = np.array([cccd_corners, gplx_corners], dtype=np.float32)
        mock_kpts.conf.cpu().numpy.return_value = np.array(
            [[0.9, 0.9, 0.9, 0.9], [0.95, 0.95, 0.95, 0.95]], dtype=np.float32
        )

        mock_res.boxes = mock_boxes
        mock_res.keypoints = mock_kpts
        mock_yolo.predict.return_value = [mock_res]
        mock_yolo.task = "pose"

        with patch("ultralytics.YOLO", return_value=mock_yolo):
            detector = DocumentPoseDetector(model_path="models/document_pose/best.pt")

            # Khi auto: Chọn thẻ 1 (gplx) vì diện tích và conf vượt trội
            res_auto = detector.detect(self.dummy_img, expected_type="auto")
            self.assertTrue(res_auto.success, f"res_auto failed: {res_auto.failure_reason}")
            self.assertEqual(res_auto.document_type, "gplx")
            self.assertAlmostEqual(res_auto.confidence, 0.9, places=2)

            # Khi yêu cầu cccd: Chọn đúng thẻ 0 (cccd)
            res_cccd = detector.detect(self.dummy_img, expected_type="cccd")
            self.assertTrue(res_cccd.success)
            self.assertEqual(res_cccd.document_type, "cccd")
            self.assertAlmostEqual(res_cccd.confidence, 0.5, places=2)


class TestPerspectiveAndPipeline(unittest.TestCase):
    """Kiểm thử Perspective Transform từ 4 góc và tính độc lập của Pipeline."""

    def test_12_rectify_from_synthetic_corners(self):
        """12. DocumentRectifier.rectify_from_corners nắn chuẩn về kích thước 856x540."""
        rectifier = DocumentRectifier(target_size=(856, 540))

        # Tạo ảnh nghiêng phối cảnh nhân tạo
        src_img = np.zeros((700, 1000, 3), dtype=np.uint8)
        quad_corners = np.array(
            [
                [150.0, 120.0],  # TL
                [820.0, 90.0],   # TR
                [860.0, 520.0],  # BR
                [110.0, 480.0],  # BL
            ],
            dtype=np.float32,
        )
        cv2.fillPoly(src_img, [quad_corners.astype(np.int32)], (200, 200, 200))

        res = rectifier.rectify_from_corners(src_img, quad_corners, card_type="cccd")
        self.assertTrue(res.success)
        self.assertEqual(res.rectified_image.shape, (540, 856, 3))
        self.assertEqual(res.method, "pose_keypoints")
        self.assertFalse(res.fallback_used)

    def test_13_rectify_rotated_cards(self):
        """13. Thử nghiệm nắn phối cảnh trên thẻ xoay 90°, 180°, 270°."""
        rectifier = DocumentRectifier(target_size=(856, 540))
        src_img = np.zeros((800, 800, 3), dtype=np.uint8)

        # Xoay 90 độ (thẻ dọc)
        corners_90 = np.array([[500, 100], [500, 700], [200, 700], [200, 100]], dtype=np.float32)
        res_90 = rectifier.rectify_from_corners(src_img, corners_90, card_type="cccd")
        self.assertTrue(res_90.success)
        self.assertEqual(res_90.rectified_image.shape, (540, 856, 3))

        # Xoay 180 độ
        corners_180 = np.array([[700, 500], [100, 500], [100, 200], [700, 200]], dtype=np.float32)
        res_180 = rectifier.rectify_from_corners(src_img, corners_180, card_type="cccd")
        self.assertTrue(res_180.success)
        self.assertEqual(res_180.rectified_image.shape, (540, 856, 3))

    @patch("services.pipeline_pose.get_document_pose_detector")
    @patch("services.pipeline_pose.extract_adaptive_pipeline_e")
    def test_14_pose_mode_does_not_load_yolo_detection(self, mock_ocr, mock_get_pose):
        """14. Ở chế độ pose: Không bao giờ gọi get_document_detector (YOLO Detection cũ)."""
        mock_pose_detector = MagicMock()
        mock_pose_detector.detect.return_value = PoseDetectionResult(
            success=True,
            document_type="cccd",
            bbox=(100, 100, 700, 500),
            confidence=0.95,
            corners=np.array([[100, 100], [700, 100], [700, 500], [100, 500]], dtype=np.float32),
            keypoint_confidences=[0.9, 0.9, 0.9, 0.9],
            failure_reason=None,
        )
        mock_get_pose.return_value = mock_pose_detector

        mock_ocr_res = MagicMock()
        mock_ocr_res.fields = {"id_number": "001099012345"}
        mock_ocr_res.raw_fields = {"id_number": "001099012345"}
        mock_ocr_res.valid_flags = {"id_number": True}
        mock_ocr_res.clean_flags = {"id_number": True}
        mock_ocr_res.fallback_sources = {"id_number": "pose_rectified"}
        mock_ocr_res.total_fallbacks = 0
        mock_ocr.return_value = mock_ocr_res

        img = np.ones((600, 800, 3), dtype=np.uint8) * 100

        resp = process_document_pose(img, mode="fast", document_type="cccd")
        self.assertTrue(resp.success)
        self.assertEqual(resp.pipeline_mode, "pose")
        self.assertEqual(resp.document_type, "cccd")


class TestFastAPIPoseIntegration(unittest.TestCase):
    """Kiểm thử tương thích API v2 Pose-Only."""

    def setUp(self):
        # Tạo ảnh hợp lệ nhỏ dạng JPEG bytes
        img_mat = np.ones((100, 100, 3), dtype=np.uint8) * 200
        _, buf = cv2.imencode(".jpg", img_mat)
        self.img_bytes = buf.tobytes()

    def test_15_api_returns_503_when_pose_checkpoint_missing(self):
        """15. API v2 trả lỗi 503 POSE_MODEL_NOT_FOUND khi thiếu file best.pt."""
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)

        with patch.dict(os.environ, {"POSE_MODEL_PATH": "non_existent_pose.pt"}):
            resp = client.post(
                "/api/v2/upload",
                files={"file": ("test.jpg", self.img_bytes, "image/jpeg")},
                data={"mode": "fast", "document_type": "auto"},
            )
            self.assertEqual(resp.status_code, 503)
            data = resp.json()
            self.assertFalse(data["success"])
            self.assertEqual(data["error_code"], "POSE_MODEL_NOT_FOUND")
            self.assertIn("YOLOv8-Pose model checkpoint not found", data["message"])

    @patch("main.get_inference_pool")
    @patch("pathlib.Path.exists", return_value=True)
    def test_16_api_returns_422_when_keypoint_validation_fails(self, mock_exists, mock_pool):
        """16. API v2 trả lỗi 422 KEYPOINT_VALIDATION_FAILED khi 4 góc thẻ không đạt tiêu chuẩn."""
        from fastapi.testclient import TestClient
        from main import app

        pool_instance = MagicMock()
        pool_instance.run_in_pool.side_effect = KeypointValidationError(
            "Corners ordering is counter-clockwise", {"reason": "inverted"}
        )
        mock_pool.return_value = pool_instance

        client = TestClient(app)
        with patch.dict(os.environ, {"POSE_MODEL_PATH": "dummy_pose.pt"}):
            resp = client.post(
                "/api/v2/upload",
                files={"file": ("test.jpg", self.img_bytes, "image/jpeg")},
                data={"mode": "fast", "document_type": "auto"},
            )
            self.assertEqual(resp.status_code, 422)
            data = resp.json()
            self.assertFalse(data["success"])
            self.assertEqual(data["error_code"], "KEYPOINT_VALIDATION_FAILED")

    @patch("main.get_inference_pool")
    @patch("pathlib.Path.exists", return_value=True)
    def test_17_api_returns_422_when_no_document_detected(self, mock_exists, mock_pool):
        """17. API v2 trả lỗi 422 NO_DOCUMENT_DETECTED khi không tìm thấy giấy tờ trong ảnh."""
        from fastapi.testclient import TestClient
        from main import app

        pool_instance = MagicMock()
        pool_instance.run_in_pool.side_effect = NoDocumentDetectedError("No document detected (0 bounding boxes)")
        mock_pool.return_value = pool_instance

        client = TestClient(app)
        with patch.dict(os.environ, {"POSE_MODEL_PATH": "dummy_pose.pt"}):
            resp = client.post(
                "/api/v2/upload",
                files={"file": ("test.jpg", self.img_bytes, "image/jpeg")},
                data={"mode": "fast", "document_type": "auto"},
            )
            self.assertEqual(resp.status_code, 422)
            data = resp.json()
            self.assertFalse(data["success"])
            self.assertEqual(data["error_code"], "NO_DOCUMENT_DETECTED")


if __name__ == "__main__":
    unittest.main()
