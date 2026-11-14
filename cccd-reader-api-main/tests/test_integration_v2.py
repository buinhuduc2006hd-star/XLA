"""Integration test cho API v2 trên mô hình thật (YOLOv8n + EasyOCR).

Giai đoạn 4: Kiểm thử tích hợp thực tế End-to-End từ tiếp nhận ảnh,
phát hiện YOLO, nắn phối cảnh, xoay hướng văn bản, đến trích xuất EasyOCR.
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from main import app
from schemas import DocumentV2Response


class TestAPIV2Integration(unittest.TestCase):
    """Kiểm thử tích hợp End-to-End với model YOLOv8n và EasyOCR thực tế."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Tìm một ảnh mẫu CCCD thực tế trong tập test
        test_dir = PROJECT_ROOT / "datasets" / "images" / "test"
        cccd_samples = list(test_dir.glob("cccd_*.jpg"))
        if not cccd_samples:
            raise unittest.SkipTest("Không tìm thấy ảnh test CCCD trong datasets/images/test")
        cls.sample_cccd_path = cccd_samples[0]

        gplx_samples = list(test_dir.glob("gplx_*.jpg"))
        cls.sample_gplx_path = gplx_samples[0] if gplx_samples else None

    def test_live_v2_upload_cccd_fast_mode(self):
        """Kiểm thử gọi trực tiếp POST /api/v2/upload với mode=fast trên ảnh CCCD thật."""
        with open(self.sample_cccd_path, "rb") as f:
            image_bytes = f.read()

        response = self.client.post(
            "/api/v2/upload",
            files={"file": (self.sample_cccd_path.name, image_bytes, "image/jpeg")},
            data={"mode": "fast", "document_type": "auto"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Kiểm tra tính tuân thủ schema Pydantic
        parsed = DocumentV2Response.model_validate(data)
        self.assertTrue(parsed.success)
        self.assertEqual(parsed.document_type, "cccd")
        self.assertEqual(parsed.mode, "fast")
        self.assertGreater(parsed.detection.confidence, 0.70)
        self.assertEqual(len(parsed.detection.bbox), 4)

        # Kiểm tra các trường đã được trích xuất
        self.assertIn("id_number", parsed.fields)
        self.assertIn("date_of_birth", parsed.fields)
        self.assertGreater(parsed.timing_ms.total_ms, 0)
        print(f"\n[Live CCCD Fast] Total latency: {parsed.timing_ms.total_ms:.1f}ms | ID: {parsed.fields['id_number'].value}")

    def test_live_v2_upload_gplx_fast_mode(self):
        """Kiểm thử gọi trực tiếp POST /api/v2/upload với mode=fast trên ảnh GPLX thật."""
        if self.sample_gplx_path is None:
            self.skipTest("Không có ảnh GPLX")

        with open(self.sample_gplx_path, "rb") as f:
            image_bytes = f.read()

        response = self.client.post(
            "/api/v2/upload",
            files={"file": (self.sample_gplx_path.name, image_bytes, "image/jpeg")},
            data={"mode": "fast", "document_type": "gplx"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        parsed = DocumentV2Response.model_validate(data)
        self.assertTrue(parsed.success)
        self.assertEqual(parsed.document_type, "gplx")
        self.assertGreater(parsed.detection.confidence, 0.70)
        self.assertIn("license_class", parsed.fields)
        print(f"\n[Live GPLX Fast] Total latency: {parsed.timing_ms.total_ms:.1f}ms | Class: {parsed.fields['license_class'].value}")


if __name__ == "__main__":
    unittest.main()

