# CCCD Reader Backend API & AI Services

Backend dịch vụ FastAPI và xử lý AI cho hệ thống nhận dạng tài liệu CCCD và GPLX.

## Cấu trúc thư mục

- `configs/`: File cấu hình dataset YOLO (`data.yaml`).
- `datasets/`: Cấu trúc thư mục dataset chuẩn YOLO (`images/`, `labels/`).
- `docs/`: Tài liệu hướng dẫn (`dataset_guide.md`).
- `scripts/`: Công cụ kiểm tra ROI (`debug_roi.py`), dự đoán thử (`predict_sample.py`) và scripts huấn luyện (`scripts/train/`).
- `services/`:
  - `document_detector.py`: Tự động phát hiện tài liệu bằng YOLOv8n.
  - `cccd_extractor.py`: Trích xuất 8 trường thông tin CCCD gắn chip.
  - `gplx_extractor.py`: Trích xuất 7 trường thông tin GPLX.
  - `id_card_extractor.py`: Phân loại và điều hướng trích xuất.
  - `ocr_service.py`: Wrapper khởi tạo EasyOCR.
- `tests/`: Bộ kiểm thử tự động toàn diện (`test_cccd_extractor.py`, `test_document_detector.py`).
- `uploads/`: Thư mục lưu tạm thời ảnh khi tải lên qua API.
- `main.py`: FastAPI server (endpoint `/api/v1/upload`).

## Hướng dẫn chạy

```bash
# Khởi động Backend API
uvicorn main:app --reload --port 8000

# Chạy toàn bộ Unit Tests
python -m unittest discover -s tests -p "test_*.py" -v
```
