"""Unit tests cho API v2 (FastAPI Integration & End-to-End OCR Pipeline).

Giai đoạn 4: Đánh giá độc lập 18 tiêu chí với mock YOLO và OCR để đảm bảo
chạy siêu nhanh (< 2s) và kiểm thử toàn diện các nhánh logic và xử lý lỗi.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from fastapi.testclient import TestClient

from main import app
from schemas import DocumentV2Response
from services.adaptive_extractor import ExtractionResult
from services.document_detector import Detection, DetectionResult
from services.document_orientation import OrientationResult
from services.document_rectifier import RectifyResult


def _create_dummy_image_bytes(width: int = 200, height: int = 150, format: str = ".jpg") -> bytes:
    """Tạo byte array của ảnh giả lập hợp lệ."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (200, 200, 200)  # Xám sáng
    # Vẽ vài chi tiết giả
    cv2.rectangle(img, (20, 20), (width - 20, height - 20), (0, 0, 0), 2)
    success, encoded = cv2.imencode(format, img)
    assert success, "imencode failed"
    return encoded.tobytes()


class TestAPIV2Unit(unittest.TestCase):
    """Bộ kiểm thử đơn vị cho POST /api/v2/upload và kiểm tra hồi quy API v1."""

    def setUp(self):
        self.client = TestClient(app)
        self.valid_jpg_bytes = _create_dummy_image_bytes(300, 200, ".jpg")
        self.valid_png_bytes = _create_dummy_image_bytes(300, 200, ".png")

    # -----------------------------------------------------------------------
    # 1. Upload CCCD hợp lệ
    # -----------------------------------------------------------------------
    @patch("services.pipeline_v2.extract_adaptive_pipeline_e")
    @patch("services.pipeline_v2.get_orientation_corrector")
    @patch("services.pipeline_v2.get_document_rectifier")
    @patch("services.pipeline_v2.get_document_detector")
    def test_upload_valid_cccd(self, mock_get_det, mock_get_rect, mock_get_orient, mock_extract):
        mock_det = MagicMock()
        mock_det.detect.return_value = DetectionResult(
            detections=[Detection(0, "cccd", 0.95, (10, 10, 290, 190))],
            image_shape=(200, 300),
            device="cpu",
            has_detection=True,
            top_detection=Detection(0, "cccd", 0.95, (10, 10, 290, 190)),
        )
        mock_get_det.return_value = mock_det

        mock_rect = MagicMock()
        mock_rect.rectify.return_value = RectifyResult(
            success=True,
            rectified_image=np.zeros((540, 856, 3), dtype=np.uint8),
            corners=[[10, 10], [290, 10], [290, 190], [10, 190]],
            corners_crop=None,
            method="contour_approx",
            confidence=0.9,
            fallback_used=False,
            error=None,
            aspect_ratio=1.58,
            rotation_deg=0.0,
        )
        mock_get_rect.return_value = mock_rect

        mock_orient = MagicMock()
        mock_orient.correct_orientation.return_value = OrientationResult(
            success=True,
            image=np.zeros((540, 856, 3), dtype=np.uint8),
            rotation_deg=0,
            method="anchor_header_strip",
            confidence=0.85,
            fallback_used=False,
            details={},
        )
        mock_get_orient.return_value = mock_orient

        mock_extract.return_value = ExtractionResult(
            fields={
                "id_number": "030206013743",
                "full_name": "NGUYEN VAN A",
                "date_of_birth": "01/01/2000",
            },
            valid_flags={"id_number": True, "full_name": True, "date_of_birth": True},
            fallback_used={"id_number": False, "full_name": False, "date_of_birth": False},
            fallback_sources={"id_number": "rectified", "full_name": "rectified", "date_of_birth": "rectified"},
        )

        response = self.client.post(
            "/api/v2/upload",
            files={"file": ("test_cccd.jpg", self.valid_jpg_bytes, "image/jpeg")},
            data={"mode": "fast", "document_type": "auto"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["document_type"], "cccd")
        self.assertEqual(data["detection"]["class_name"], "cccd")
        self.assertEqual(data["fields"]["id_number"]["value"], "030206013743")
        self.assertTrue(data["fields"]["id_number"]["valid"])
        self.assertIn("timing_ms", data)
        self.assertGreater(data["timing_ms"]["total_ms"], 0)

    # -----------------------------------------------------------------------
    # 2. Upload GPLX hợp lệ
    # -----------------------------------------------------------------------
    @patch("services.pipeline_v2.extract_adaptive_pipeline_e")
    @patch("services.pipeline_v2.get_orientation_corrector")
    @patch("services.pipeline_v2.get_document_rectifier")
    @patch("services.pipeline_v2.get_document_detector")
    def test_upload_valid_gplx(self, mock_get_det, mock_get_rect, mock_get_orient, mock_extract):
        mock_det = MagicMock()
        mock_det.detect.return_value = DetectionResult(
            detections=[Detection(1, "gplx", 0.92, (10, 10, 290, 190))],
            image_shape=(200, 300),
            device="cpu",
            has_detection=True,
            top_detection=Detection(1, "gplx", 0.92, (10, 10, 290, 190)),
        )
        mock_get_det.return_value = mock_det

        mock_rect = MagicMock()
        mock_rect.rectify.return_value = RectifyResult(
            success=True,
            rectified_image=np.zeros((540, 856, 3), dtype=np.uint8),
            corners=[[10, 10], [290, 10], [290, 190], [10, 190]],
            corners_crop=None,
            method="contour_approx",
            confidence=0.88,
            fallback_used=False,
            error=None,
            aspect_ratio=1.58,
            rotation_deg=0.0,
        )
        mock_get_rect.return_value = mock_rect

        mock_orient = MagicMock()
        mock_orient.correct_orientation.return_value = OrientationResult(
            success=True,
            image=np.zeros((540, 856, 3), dtype=np.uint8),
            rotation_deg=0,
            method="anchor_header_strip",
            confidence=0.8,
            fallback_used=False,
            details={},
        )
        mock_get_orient.return_value = mock_orient

        mock_extract.return_value = ExtractionResult(
            fields={"license_number": "123456789012", "license_class": "A1"},
            valid_flags={"license_number": True, "license_class": True},
            fallback_used={"license_number": False, "license_class": False},
            fallback_sources={"license_number": "rectified", "license_class": "rectified"},
        )

        response = self.client.post(
            "/api/v2/upload",
            files={"file": ("test_gplx.png", self.valid_png_bytes, "image/png")},
            data={"mode": "accurate", "document_type": "gplx"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_type"], "gplx")
        self.assertEqual(data["mode"], "accurate")
        self.assertEqual(data["fields"]["license_class"]["value"], "A1")

    # -----------------------------------------------------------------------
    # 3. Tự động xác định loại giấy tờ (auto)
    # -----------------------------------------------------------------------
    @patch("services.pipeline_v2.extract_adaptive_pipeline_e")
    @patch("services.pipeline_v2.get_orientation_corrector")
    @patch("services.pipeline_v2.get_document_rectifier")
    @patch("services.pipeline_v2.get_document_detector")
    def test_auto_document_type(self, mock_get_det, mock_get_rect, mock_get_orient, mock_extract):
        mock_det = MagicMock()
        mock_det.detect.return_value = DetectionResult(
            detections=[Detection(1, "gplx", 0.89, (5, 5, 295, 195))],
            image_shape=(200, 300),
            device="cpu",
            has_detection=True,
            top_detection=Detection(1, "gplx", 0.89, (5, 5, 295, 195)),
        )
        mock_get_det.return_value = mock_det

        mock_rect = MagicMock()
        mock_rect.rectify.return_value = RectifyResult(
            success=True, rectified_image=np.zeros((540, 856, 3), dtype=np.uint8),
            corners=None, corners_crop=None, method="fallback_crop", confidence=0.5,
            fallback_used=True, error=None, aspect_ratio=1.58, rotation_deg=0.0
        )
        mock_get_rect.return_value = mock_rect

        mock_orient = MagicMock()
        mock_orient.correct_orientation.return_value = OrientationResult(
            success=True, image=np.zeros((540, 856, 3), dtype=np.uint8),
            rotation_deg=0, method="anchor", confidence=0.7, fallback_used=False, details={}
        )
        mock_get_orient.return_value = mock_orient

        mock_extract.return_value = ExtractionResult(fields={}, valid_flags={}, fallback_used={}, fallback_sources={})

        response = self.client.post(
            "/api/v2/upload",
            files={"file": ("auto_sample.jpg", self.valid_jpg_bytes, "image/jpeg")},
            data={"document_type": "auto"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document_type"], "gplx")

    # -----------------------------------------------------------------------
    # 4. Mode fast vs accurate
    # -----------------------------------------------------------------------
    @patch("services.pipeline_v2.extract_adaptive_pipeline_e")
    @patch("services.pipeline_v2.get_orientation_corrector")
    @patch("services.pipeline_v2.get_document_rectifier")
    @patch("services.pipeline_v2.get_document_detector")
    def test_mode_handling(self, mock_get_det, mock_get_rect, mock_get_orient, mock_extract):
        mock_det = MagicMock()
        mock_det.detect.return_value = DetectionResult(
            detections=[Detection(0, "cccd", 0.9, (0, 0, 100, 100))],
            image_shape=(100, 100), device="cpu", has_detection=True,
            top_detection=Detection(0, "cccd", 0.9, (0, 0, 100, 100))
        )
        mock_get_det.return_value = mock_det
        mock_get_rect.return_value.rectify.return_value = RectifyResult(
            success=True, rectified_image=np.zeros((540, 856, 3), dtype=np.uint8),
            corners=None, corners_crop=None, method="contour", confidence=0.8,
            fallback_used=False, error=None, aspect_ratio=1.58, rotation_deg=0.0
        )
        mock_get_orient.return_value.correct_orientation.return_value = OrientationResult(
            success=True, image=np.zeros((540, 856, 3), dtype=np.uint8),
            rotation_deg=0, method="anchor", confidence=0.8, fallback_used=False, details={}
        )
        mock_extract.return_value = ExtractionResult(fields={}, valid_flags={}, fallback_used={}, fallback_sources={})

        # Test mode = fast
        res_fast = self.client.post(
            "/api/v2/upload",
            files={"file": ("sample.jpg", self.valid_jpg_bytes, "image/jpeg")},
            data={"mode": "fast"},
        )
        self.assertEqual(res_fast.status_code, 200)
        self.assertEqual(res_fast.json()["mode"], "fast")
        mock_extract.assert_called_with(rectified_image=mock_get_orient.return_value.correct_orientation.return_value.image, yolo_crop=unittest.mock.ANY, card_type="cccd", mode="fast")

        # Test mode = accurate
        res_acc = self.client.post(
            "/api/v2/upload",
            files={"file": ("sample.jpg", self.valid_jpg_bytes, "image/jpeg")},
            data={"mode": "accurate"},
        )
        self.assertEqual(res_acc.status_code, 200)
        self.assertEqual(res_acc.json()["mode"], "accurate")
        mock_extract.assert_called_with(rectified_image=mock_get_orient.return_value.correct_orientation.return_value.image, yolo_crop=unittest.mock.ANY, card_type="cccd", mode="accurate")

    # -----------------------------------------------------------------------
    # 5. Không tìm thấy giấy tờ (No detection -> 422)
    # -----------------------------------------------------------------------
    @patch("services.pipeline_v2.get_document_detector")
    def test_no_detection(self, mock_get_det):
        mock_det = MagicMock()
        mock_det.detect.return_value = DetectionResult(
            detections=[], image_shape=(200, 300), device="cpu",
            has_detection=False, top_detection=None
        )
        mock_get_det.return_value = mock_det

        response = self.client.post(
            "/api/v2/upload",
            files={"file": ("blank.jpg", self.valid_jpg_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error_code"], "NO_DOCUMENT_DETECTED")

    # -----------------------------------------------------------------------
    # 6. Multiple detections (chọn top detection)
    # -----------------------------------------------------------------------
    @patch("services.pipeline_v2.extract_adaptive_pipeline_e")
    @patch("services.pipeline_v2.get_orientation_corrector")
    @patch("services.pipeline_v2.get_document_rectifier")
    @patch("services.pipeline_v2.get_document_detector")
    def test_multiple_detections(self, mock_get_det, mock_get_rect, mock_get_orient, mock_extract):
        mock_det = MagicMock()
        det1 = Detection(0, "cccd", 0.95, (10, 10, 200, 150))
        det2 = Detection(1, "gplx", 0.60, (200, 10, 290, 150))
        mock_det.detect.return_value = DetectionResult(
            detections=[det1, det2], image_shape=(200, 300), device="cpu",
            has_detection=True, top_detection=det1
        )
        mock_get_det.return_value = mock_det

        mock_get_rect.return_value.rectify.return_value = RectifyResult(
            success=True, rectified_image=np.zeros((540, 856, 3), dtype=np.uint8),
            corners=None, corners_crop=None, method="contour", confidence=0.9,
            fallback_used=False, error=None, aspect_ratio=1.58, rotation_deg=0.0
        )
        mock_get_orient.return_value.correct_orientation.return_value = OrientationResult(
            success=True, image=np.zeros((540, 856, 3), dtype=np.uint8),
            rotation_deg=0, method="anchor", confidence=0.8, fallback_used=False, details={}
        )
        mock_extract.return_value = ExtractionResult(fields={}, valid_flags={}, fallback_used={}, fallback_sources={})

        response = self.client.post(
            "/api/v2/upload",
            files={"file": ("multi.jpg", self.valid_jpg_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["detection"]["num_detections"], 2)
        self.assertEqual(data["document_type"], "cccd")
        self.assertEqual(data["detection"]["confidence"], 0.95)

    # -----------------------------------------------------------------------
    # 7. Document type mismatch (phát hiện cccd nhưng người dùng yêu cầu gplx)
    # -----------------------------------------------------------------------
    @patch("services.pipeline_v2.get_document_detector")
    def test_document_type_mismatch(self, mock_get_det):
        mock_det = MagicMock()
        mock_det.detect.return_value = DetectionResult(
            detections=[Detection(0, "cccd", 0.95, (10, 10, 290, 190))],
            image_shape=(200, 300), device="cpu", has_detection=True,
            top_detection=Detection(0, "cccd", 0.95, (10, 10, 290, 190))
        )
        mock_get_det.return_value = mock_det

        response = self.client.post(
            "/api/v2/upload",
            files={"file": ("sample.jpg", self.valid_jpg_bytes, "image/jpeg")},
            data={"document_type": "gplx"},
        )
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error_code"], "DOCUMENT_TYPE_MISMATCH")

    # -----------------------------------------------------------------------
    # 8. Rectification fallback
    # -----------------------------------------------------------------------
    @patch("services.pipeline_v2.extract_adaptive_pipeline_e")
    @patch("services.pipeline_v2.get_orientation_corrector")
    @patch("services.pipeline_v2.get_document_rectifier")
    @patch("services.pipeline_v2.get_document_detector")
    def test_rectification_fallback(self, mock_get_det, mock_get_rect, mock_get_orient, mock_extract):
        mock_det = MagicMock()
        mock_det.detect.return_value = DetectionResult(
            detections=[Detection(0, "cccd", 0.8, (10, 10, 200, 150))],
            image_shape=(200, 300), device="cpu", has_detection=True,
            top_detection=Detection(0, "cccd", 0.8, (10, 10, 200, 150))
        )
        mock_get_det.return_value = mock_det

        mock_get_rect.return_value.rectify.return_value = RectifyResult(
            success=False, rectified_image=np.zeros((540, 856, 3), dtype=np.uint8),
            corners=None, corners_crop=None, method="fallback_crop", confidence=0.0,
            fallback_used=True, error="No 4 corners", aspect_ratio=1.58, rotation_deg=0.0
        )
        mock_get_orient.return_value.correct_orientation.return_value = OrientationResult(
            success=True, image=np.zeros((540, 856, 3), dtype=np.uint8),
            rotation_deg=0, method="anchor", confidence=0.8, fallback_used=False, details={}
        )
        mock_extract.return_value = ExtractionResult(fields={}, valid_flags={}, fallback_used={}, fallback_sources={})

        response = self.client.post(
            "/api/v2/upload",
            files={"file": ("rect_fail.jpg", self.valid_jpg_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["processing"]["rectification_success"])
        self.assertTrue(data["processing"]["fallback_used"])

    # -----------------------------------------------------------------------
    # 9. Orientation fallback
    # -----------------------------------------------------------------------
    @patch("services.pipeline_v2.extract_adaptive_pipeline_e")
    @patch("services.pipeline_v2.get_orientation_corrector")
    @patch("services.pipeline_v2.get_document_rectifier")
    @patch("services.pipeline_v2.get_document_detector")
    def test_orientation_fallback(self, mock_get_det, mock_get_rect, mock_get_orient, mock_extract):
        mock_det = MagicMock()
        mock_det.detect.return_value = DetectionResult(
            detections=[Detection(0, "cccd", 0.8, (10, 10, 200, 150))],
            image_shape=(200, 300), device="cpu", has_detection=True,
            top_detection=Detection(0, "cccd", 0.8, (10, 10, 200, 150))
        )
        mock_get_det.return_value = mock_det

        mock_get_rect.return_value.rectify.return_value = RectifyResult(
            success=True, rectified_image=np.zeros((540, 856, 3), dtype=np.uint8),
            corners=None, corners_crop=None, method="contour", confidence=0.8,
            fallback_used=False, error=None, aspect_ratio=1.58, rotation_deg=0.0
        )
        mock_get_orient.return_value.correct_orientation.return_value = OrientationResult(
            success=False, image=np.zeros((540, 856, 3), dtype=np.uint8),
            rotation_deg=0, method="fallback_low_confidence", confidence=0.1,
            fallback_used=True, details={}
        )
        mock_extract.return_value = ExtractionResult(fields={}, valid_flags={}, fallback_used={}, fallback_sources={})

        response = self.client.post(
            "/api/v2/upload",
            files={"file": ("orient_fail.jpg", self.valid_jpg_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["processing"]["orientation_fallback"])
        self.assertEqual(data["processing"]["orientation_degrees"], 0)

    # -----------------------------------------------------------------------
    # 10. OCR trả về trường không hợp lệ
    # -----------------------------------------------------------------------
    @patch("services.pipeline_v2.extract_adaptive_pipeline_e")
    @patch("services.pipeline_v2.get_orientation_corrector")
    @patch("services.pipeline_v2.get_document_rectifier")
    @patch("services.pipeline_v2.get_document_detector")
    def test_ocr_invalid_fields(self, mock_get_det, mock_get_rect, mock_get_orient, mock_extract):
        mock_det = MagicMock()
        mock_det.detect.return_value = DetectionResult(
            detections=[Detection(0, "cccd", 0.8, (10, 10, 200, 150))],
            image_shape=(200, 300), device="cpu", has_detection=True,
            top_detection=Detection(0, "cccd", 0.8, (10, 10, 200, 150))
        )
        mock_get_det.return_value = mock_det
        mock_get_rect.return_value.rectify.return_value = RectifyResult(
            success=True, rectified_image=np.zeros((540, 856, 3), dtype=np.uint8),
            corners=None, corners_crop=None, method="contour", confidence=0.8,
            fallback_used=False, error=None, aspect_ratio=1.58, rotation_deg=0.0
        )
        mock_get_orient.return_value.correct_orientation.return_value = OrientationResult(
            success=True, image=np.zeros((540, 856, 3), dtype=np.uint8),
            rotation_deg=0, method="anchor", confidence=0.8, fallback_used=False, details={}
        )
        mock_extract.return_value = ExtractionResult(
            fields={"id_number": "123", "full_name": None},
            valid_flags={"id_number": False, "full_name": False},
            fallback_used={"id_number": False, "full_name": False},
            fallback_sources={"id_number": "rectified", "full_name": "unextracted"},
        )

        response = self.client.post(
            "/api/v2/upload",
            files={"file": ("invalid_fields.jpg", self.valid_jpg_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        fields = response.json()["fields"]
        self.assertEqual(fields["id_number"]["value"], "123")
        self.assertFalse(fields["id_number"]["valid"])
        self.assertIsNone(fields["full_name"]["value"])
        self.assertFalse(fields["full_name"]["valid"])

    # -----------------------------------------------------------------------
    # 11. File không phải ảnh (.txt) -> 400
    # -----------------------------------------------------------------------
    def test_file_not_image(self):
        response = self.client.post(
            "/api/v2/upload",
            files={"file": ("document.txt", b"plain text content", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error_code"], "UNSUPPORTED_MEDIA_TYPE")

    # -----------------------------------------------------------------------
    # 12. File quá lớn (> 15MB) -> 413
    # -----------------------------------------------------------------------
    def test_file_too_large(self):
        large_payload = b"0" * (16 * 1024 * 1024)  # 16MB
        response = self.client.post(
            "/api/v2/upload",
            files={"file": ("huge.jpg", large_payload, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error_code"], "FILE_TOO_LARGE")

    # -----------------------------------------------------------------------
    # 13. Payload rỗng (0 bytes) -> 400
    # -----------------------------------------------------------------------
    def test_empty_file(self):
        response = self.client.post(
            "/api/v2/upload",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error_code"], "EMPTY_FILE")

    # -----------------------------------------------------------------------
    # 14. Ảnh hỏng / không giải mã được -> 400
    # -----------------------------------------------------------------------
    def test_corrupt_image(self):
        corrupt_bytes = b"not_a_valid_image_header_bytes_12345"
        response = self.client.post(
            "/api/v2/upload",
            files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error_code"], "INVALID_IMAGE")

    # -----------------------------------------------------------------------
    # 15. OCR runtime error -> 500 (không leak stack trace)
    # -----------------------------------------------------------------------
    @patch("services.pipeline_v2.extract_adaptive_pipeline_e")
    @patch("services.pipeline_v2.get_orientation_corrector")
    @patch("services.pipeline_v2.get_document_rectifier")
    @patch("services.pipeline_v2.get_document_detector")
    def test_ocr_runtime_error(self, mock_get_det, mock_get_rect, mock_get_orient, mock_extract):
        mock_det = MagicMock()
        mock_det.detect.return_value = DetectionResult(
            detections=[Detection(0, "cccd", 0.9, (0, 0, 100, 100))],
            image_shape=(100, 100), device="cpu", has_detection=True,
            top_detection=Detection(0, "cccd", 0.9, (0, 0, 100, 100))
        )
        mock_get_det.return_value = mock_det
        mock_get_rect.return_value.rectify.return_value = RectifyResult(
            success=True, rectified_image=np.zeros((540, 856, 3), dtype=np.uint8),
            corners=None, corners_crop=None, method="contour", confidence=0.8,
            fallback_used=False, error=None, aspect_ratio=1.58, rotation_deg=0.0
        )
        mock_get_orient.return_value.correct_orientation.return_value = OrientationResult(
            success=True, image=np.zeros((540, 856, 3), dtype=np.uint8),
            rotation_deg=0, method="anchor", confidence=0.8, fallback_used=False, details={}
        )
        # Bắn lỗi RuntimeError trong OCR
        mock_extract.side_effect = RuntimeError("Internal CUDA/CPU crash /path/to/private/file.py")

        response = self.client.post(
            "/api/v2/upload",
            files={"file": ("crash.jpg", self.valid_jpg_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error_code"], "INTERNAL_SERVER_ERROR")
        # Đảm bảo không lộ path nội bộ
        self.assertNotIn("/path/to/private", str(data))

    # -----------------------------------------------------------------------
    # 16. Phản hồi tuân thủ nghiêm ngặt Pydantic Schema
    # -----------------------------------------------------------------------
    @patch("services.pipeline_v2.extract_adaptive_pipeline_e")
    @patch("services.pipeline_v2.get_orientation_corrector")
    @patch("services.pipeline_v2.get_document_rectifier")
    @patch("services.pipeline_v2.get_document_detector")
    def test_schema_compliance(self, mock_get_det, mock_get_rect, mock_get_orient, mock_extract):
        mock_det = MagicMock()
        mock_det.detect.return_value = DetectionResult(
            detections=[Detection(0, "cccd", 0.95, (10, 10, 200, 150))],
            image_shape=(200, 300), device="cpu", has_detection=True,
            top_detection=Detection(0, "cccd", 0.95, (10, 10, 200, 150))
        )
        mock_get_det.return_value = mock_det
        mock_get_rect.return_value.rectify.return_value = RectifyResult(
            success=True, rectified_image=np.zeros((540, 856, 3), dtype=np.uint8),
            corners=[[10, 10], [200, 10], [200, 150], [10, 150]], corners_crop=None,
            method="contour_approx", confidence=0.9, fallback_used=False, error=None,
            aspect_ratio=1.58, rotation_deg=0.0
        )
        mock_get_orient.return_value.correct_orientation.return_value = OrientationResult(
            success=True, image=np.zeros((540, 856, 3), dtype=np.uint8),
            rotation_deg=0, method="anchor_header_strip", confidence=0.85, fallback_used=False, details={}
        )
        mock_extract.return_value = ExtractionResult(
            fields={"id_number": "030206013743"},
            valid_flags={"id_number": True},
            fallback_used={"id_number": False},
            fallback_sources={"id_number": "rectified"},
        )

        response = self.client.post(
            "/api/v2/upload",
            files={"file": ("test.jpg", self.valid_jpg_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        # Parse lại qua DocumentV2Response để xác minh tính tương thích 100%
        parsed = DocumentV2Response.model_validate(response.json())
        self.assertEqual(parsed.document_type, "cccd")
        self.assertEqual(parsed.fields["id_number"].value, "030206013743")

    # -----------------------------------------------------------------------
    # 17. API v1 không bị ảnh hưởng (Regression Check)
    # -----------------------------------------------------------------------
    @patch("main.extract_id_card_fields")
    def test_api_v1_regression(self, mock_extract_v1):
        mock_extract_v1.return_value = {
            "card_type": "cccd",
            "id_number": "030206013743",
            "name": "NGUYEN VAN A",
        }

        response = self.client.post(
            "/api/v1/upload",
            files={"file": ("sample_v1.jpg", self.valid_jpg_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["message"], "CCCD extraction completed successfully")
        self.assertIn("data", data)
        self.assertEqual(data["data"]["card_type"], "cccd")
        self.assertEqual(data["data"]["id_number"], "030206013743")


if __name__ == "__main__":
    unittest.main()
