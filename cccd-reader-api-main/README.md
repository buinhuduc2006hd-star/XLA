# CCCD & GPLX Reader Backend API (FastAPI & AI Services)

Hệ thống Backend API và dịch vụ thị giác máy tính / học sâu phục vụ nhận dạng, nắn chỉnh phối cảnh và trích xuất thông tin tự động từ ảnh chụp Căn cước công dân (CCCD) và Giấy phép lái xe (GPLX) Việt Nam.

---

## 1. Kiến trúc xử lý Pipeline V2 (End-to-End)

```
[Ảnh người dùng tải lên (JPG, PNG, WEBP)]
                 │
                 ▼
     [Kiểm tra & Phòng vệ an toàn]
       • Định dạng ảnh hợp lệ (Magic bytes / Decoder)
       • Kích thước tối đa 15MB
       • Giải mã in-memory trực tiếp trong RAM (không ghi file tạm ra đĩa)
       • Chống Decompression Bomb (giới hạn tối đa 36 Megapixels)
                 │
                 ▼
     [YOLOv8n Document Detection] (Tái sử dụng Singleton Model)
       • Phát hiện vị trí bounding box & phân loại: Class 0 = cccd, Class 1 = gplx
       • Trả về confidence, bounding box và số lượng tài liệu phát hiện được
                 │
                 ▼
     [OpenCV Document Rectifier]
       • Mở rộng 8% Safety Margin bao quát trọn vẹn viền thẻ
       • Canny Edges + Contour Approx + Convex Hull tìm 4 góc vật lý
       • Biến đổi phối cảnh (Perspective Transform) về chuẩn thẻ ID-1: 856 × 540 px
       • Fallback an toàn về YOLO crop nếu không tìm thấy 4 góc tin cậy
                 │
                 ▼
     [Document Orientation Correction]
       • Quét dải tiêu đề quốc hiệu (Anchor Header Strip: top 28%)
       • Khớp từ khóa quốc hiệu không dấu giữa các góc [0°, 90°, 180°, 270°]
       • Tự động xoay ảnh về đúng chiều đọc ngang chuẩn
                 │
                 ▼
     [Adaptive EasyOCR & Calibrated ROIs]
       • Cắt vùng thông tin theo bộ tọa độ hiệu chuẩn thực nghiệm trên ảnh 856×540
       • Tiền xử lý chuyên biệt: Tách kênh Green cho số GPLX đỏ, khử nhiễu ma trận chấm cho hạng bằng
       • Cơ chế Smart Early-Exit: Dừng ngay khi đạt tiêu chuẩn validator trường
       • Chế độ 'fast': 1 lượt xử lý ưu tiên tốc độ CPU
       • Chế độ 'accurate': Đầy đủ biến thể tiền xử lý + Bounded Fallback (thử lại từ YOLO crop)
                 │
                 ▼
     [Field Validation & JSON Response Structuring]
       • 15 bộ Validator nghiệp vụ độc lập (kiểm tra 12 số CCCD, định dạng ngày tháng, lịch hợp lệ, hạng GPLX...)
       • Trả về JSON Response phân cấp chuẩn xác
```

---

## 2. Cấu trúc thư mục

```text
cccd-reader-api-main/
├── configs/
│   └── data.yaml                   # Cấu hình dataset YOLOv8
├── datasets/
│   ├── ground_truth_test_sample.json # Tập nhãn chuẩn Ground Truth 45 trường đối chứng
│   └── images/test/                # Tập ảnh kiểm thử thực tế
├── models/
│   └── yolov8n_document.pt         # Trọng số YOLOv8n đã huấn luyện cho CCCD và GPLX
├── schemas.py                      # Pydantic schemas cho API v2 (Request, Response, Error)
├── services/
│   ├── adaptive_extractor.py       # Pipeline E, 15 Field Validators, CER/WER, Field Fallback
│   ├── calibrated_rois.py          # Tọa độ ROI hiệu chuẩn trên ảnh 856x540, dot-matrix, green channel
│   ├── cccd_extractor.py           # Baseline extractor CCCD (Giai đoạn 1)
│   ├── document_detector.py        # Wrapper phát hiện giấy tờ YOLOv8n
│   ├── document_orientation.py     # Hiệu chỉnh chiều đọc văn bản (0°, 90°, 180°, 270°)
│   ├── document_rectifier.py       # Nắn chỉnh phối cảnh 4 góc OpenCV
│   ├── gplx_extractor.py           # Baseline extractor GPLX (Giai đoạn 1)
│   ├── id_card_extractor.py        # Bộ điều hướng phân loại thẻ cũ cho API v1
│   ├── ocr_service.py              # EasyOCR wrapper và singleton reader
│   └── pipeline_v2.py              # Orchestration service điều phối toàn bộ luồng V2
├── tests/
│   ├── test_api_v2.py              # Unit tests API v2 với mock (17 test cases, chạy < 1s)
│   ├── test_integration_v2.py      # Integration tests API v2 trên model thật (CCCD & GPLX)
│   ├── test_cccd_extractor.py      # Unit tests baseline CCCD (33 test cases)
│   ├── test_document_detector.py   # Unit tests YOLO detector (14 test cases)
│   ├── test_document_rectifier.py  # Unit tests OpenCV rectifier (15 test cases)
│   ├── test_document_orientation.py# Unit tests xoay văn bản (10 test cases)
│   └── test_stage3_2_accuracy.py   # Unit tests validator và đo lường Levenshtein (15 test cases)
├── scripts/
│   ├── benchmark_v2.py             # Benchmark so sánh API v1 vs API v2 Fast vs API v2 Accurate
│   └── benchmark_stage3_2.py       # Benchmark 5 Pipeline đối sánh Ground Truth
├── uploads/                        # Thư mục lưu file tạm thời cho API v1 (được dọn sạch sau khi OCR)
├── main.py                         # FastAPI server (cung cấp cả /api/v1/upload và /api/v2/upload)
└── requirements.txt                # Danh sách thư viện phụ thuộc
```

---

## 3. Hướng dẫn khởi động Backend

Mở PowerShell tại thư mục `C:\Users\bui nhu duc\XLA`:

```powershell
# Kích hoạt môi trường ảo
& ".\.venv\Scripts\Activate.ps1"

# Di chuyển vào thư mục backend
cd "cccd-reader-api-main"

# Khởi chạy server FastAPI bằng Uvicorn
$env:PYTHONPATH = "C:\Users\bui nhu duc\XLA\cccd-reader-api-main"
& "..\.venv\Scripts\uvicorn.exe" main:app --host 0.0.0.0 --port 8000 --reload
```

Sau khi khởi động thành công:
- **API URL**: `http://localhost:8000`
- **Tài liệu Swagger UI tương tác**: `http://localhost:8000/docs`
- **Tài liệu ReDoc**: `http://localhost:8000/redoc`
- **Health Check Endpoint**: `GET http://localhost:8000/health`

---

## 4. Chi tiết Endpoint `POST /api/v2/upload`

### 4.1. Tham số đầu vào (multipart/form-data)

| Tham số | Kiểu | Bắt buộc | Mặc định | Mô tả |
|---|---|:---:|:---:|---|
| `file` | Binary | **Có** | - | File ảnh giấy tờ (`.jpg`, `.jpeg`, `.png`, `.webp`). Tối đa 15MB. |
| `mode` | String | Không | `fast` | Chế độ OCR: `fast` (nhanh, 1 pass, ~20s) hoặc `accurate` (tối ưu chính xác, adaptive prep, fallback retry, ~70-85s). |
| `document_type` | String | Không | `auto` | Bộ lọc loại thẻ: `auto` (tự động phân loại), `cccd`, hoặc `gplx`. |

### 4.2. So sánh Chế độ `fast` và `accurate`

- **Chế độ `fast` (Khuyến nghị cho ứng dụng tương tác người dùng)**:
  - Chỉ quét góc 0° và 180° ở bước định hướng.
  - Sử dụng 1 lượt tiền xử lý ưu tiên trên từng vùng ROI với cơ chế Early Exit.
  - Không kích hoạt retry/fallback từ crop gốc khi trường thiếu.
  - Tốc độ xử lý trên CPU: **~19 – 25 giây / ảnh**.
  - Duy trì độ chính xác cao cho các ảnh chụp rõ nét, không bị nghiêng lệch quá mức.

- **Chế độ `accurate` (Khuyến nghị cho xử lý hậu kiểm, ảnh khó, ảnh chụp xiên/mờ)**:
  - Kiểm tra toàn diện 4 góc xoay [0°, 90°, 180°, 270°] nếu ảnh đứng.
  - Áp dụng đầy đủ các biến thể tiền xử lý thích ứng (Original, CLAHE, Otsu Binarization, Green Channel, Dot-Matrix).
  - Tự động kích hoạt cơ chế Field-Level Fallback (tối đa 1 lần retry/trường) phục hồi từ ảnh crop gốc khi phát hiện trường bị lỗi hoặc sai format.
  - Tốc độ xử lý trên CPU: **~70 – 87 giây / ảnh**.
  - Đạt tỷ lệ trích xuất tối đa (**91.11%**) và tỷ lệ hợp lệ cao nhất (**86.67%**).

---

## 5. Ví dụ gọi API

### A. Gọi bằng PowerShell (`Invoke-RestMethod`)

```powershell
$filePath = "C:\Users\bui nhu duc\XLA\cccd-reader-api-main\datasets\images\test\cccd_sample.jpg"

# Gọi API v2 với chế độ Fast
$response = Invoke-RestMethod -Uri "http://localhost:8000/api/v2/upload" `
    -Method Post `
    -Form @{
        file = Get-Item $filePath
        mode = "fast"
        document_type = "auto"
    }

# In kết quả
$response | ConvertTo-Json -Depth 5
```

### B. Gọi bằng Python (`requests`)

```python
import requests

url = "http://localhost:8000/api/v2/upload"
file_path = "path/to/your/image.jpg"

with open(file_path, "rb") as f:
    files = {"file": ("document.jpg", f, "image/jpeg")}
    data = {
        "mode": "accurate",        # 'fast' hoặc 'accurate'
        "document_type": "auto"   # 'auto', 'cccd', hoặc 'gplx'
    }
    # Khuyến nghị thiết lập timeout client phù hợp với CPU: tối thiểu 120s cho accurate
    response = requests.post(url, files=files, data=data, timeout=150)

if response.status_code == 200:
    result = response.json()
    print("Loại thẻ:", result["document_type"])
    print("Thời gian xử lý:", result["timing_ms"]["total_ms"], "ms")
    for field_name, field_info in result["fields"].items():
        print(f" - {field_name}: {field_info['value']} (Hợp lệ: {field_info['valid']})")
else:
    print("Lỗi:", response.status_code, response.json())
```

---

## 6. Cấu trúc JSON Response mẫu (Dữ liệu đã che thông tin nhạy cảm)

```json
{
  "success": true,
  "document_type": "cccd",
  "mode": "fast",
  "detection": {
    "confidence": 0.9542,
    "bbox": [108, 92, 896, 584],
    "num_detections": 1,
    "class_name": "cccd"
  },
  "processing": {
    "rectification_success": true,
    "rectification_method": "contour_approx",
    "orientation_degrees": 0,
    "orientation_fallback": false,
    "fallback_used": false
  },
  "fields": {
    "id_number": {
      "value": "030206013743",
      "valid": true,
      "confidence": null,
      "source": "rectified"
    },
    "full_name": {
      "value": "NGUYEN VAN A",
      "valid": true,
      "confidence": null,
      "source": "rectified"
    },
    "date_of_birth": {
      "value": "27/03/2006",
      "valid": true,
      "confidence": null,
      "source": "rectified"
    },
    "gender": {
      "value": "Nam",
      "valid": true,
      "confidence": null,
      "source": "rectified"
    },
    "nationality": {
      "value": "Việt Nam",
      "valid": true,
      "confidence": null,
      "source": "rectified"
    },
    "place_of_origin": {
      "value": "Nam Chính, Nam Sách, Hải Dương",
      "valid": true,
      "confidence": null,
      "source": "rectified"
    },
    "residence": {
      "value": "Thôn Bích Đông, Nam Chính, Nam Sách, Hải Dương",
      "valid": true,
      "confidence": null,
      "source": "rectified"
    },
    "expiry_date": {
      "value": "27/03/2031",
      "valid": true,
      "confidence": null,
      "source": "rectified"
    }
  },
  "timing_ms": {
    "detection_ms": 78.4,
    "rectification_ms": 32.1,
    "orientation_ms": 1845.6,
    "ocr_ms": 22616.6,
    "total_ms": 24572.7
  }
}
```

---

## 7. Bảng mã lỗi có cấu trúc (Structured Error Codes)

Khi có sự cố xảy ra, API trả về JSON chuẩn theo schema `ErrorResponse` kèm HTTP status code tương ứng:

| HTTP Status | Mã lỗi (`error_code`) | Ý nghĩa & Nguyên nhân |
|---|---|---|
| **400** | `UNSUPPORTED_MEDIA_TYPE` | File không thuộc định dạng được hỗ trợ (chỉ nhận `.jpg`, `.jpeg`, `.png`, `.webp`). |
| **400** | `EMPTY_FILE` | Nội dung file tải lên bị rỗng (0 bytes). |
| **400** | `INVALID_IMAGE` | Dữ liệu file bị lỗi hoặc không thể giải mã thành ma trận điểm ảnh OpenCV. |
| **400** | `IMAGE_TOO_SMALL` | Kích thước ảnh quá nhỏ (nhỏ hơn 10×10 pixels). |
| **400** | `IMAGE_TOO_LARGE` | Độ phân giải ảnh vượt ngưỡng an toàn (lớn hơn 36 Megapixels). |
| **400** | `INVALID_PARAMETER` | Tham số `mode` hoặc `document_type` không nằm trong danh mục cho phép. |
| **413** | `FILE_TOO_LARGE` | Dung lượng file vượt quá giới hạn 15 Megabytes. |
| **422** | `NO_DOCUMENT_DETECTED` | Mô hình YOLOv8n không phát hiện được bất kỳ giấy tờ CCCD hoặc GPLX nào trong ảnh. |
| **422** | `DOCUMENT_TYPE_MISMATCH` | Phát hiện giấy tờ nhưng loại phát hiện được không khớp với yêu cầu của người dùng (vd: ảnh CCCD nhưng tham số yêu cầu `gplx`). |
| **500** | `INTERNAL_SERVER_ERROR` | Lỗi ngoại lệ nội bộ server (đã được che giấu stack trace và thông tin nhạy cảm). |

Ví dụ JSON lỗi:
```json
{
  "success": false,
  "error_code": "NO_DOCUMENT_DETECTED",
  "message": "No CCCD or GPLX detected in the uploaded image",
  "details": {
    "num_detections": 0,
    "image_shape": [400, 600]
  }
}
```

---

## 8. Hướng dẫn cấu hình Client Timeout trên môi trường CPU

Do hệ thống hiện tại chạy suy luận hoàn toàn trên **CPU (PyTorch CPU)**:
- **Mode `fast`**: Thời gian xử lý trung bình dao động từ **~19s đến 25s**. Phía Client nên đặt `timeout` tối thiểu **40 - 50 giây**.
- **Mode `accurate`**: Thời gian xử lý dao động từ **~70s đến 87s** (phân vị 95% có thể lên tới 140s đối với ảnh có nhiều chữ hoặc kích hoạt fallback). Phía Client nên đặt `timeout` tối thiểu **120 - 150 giây**.
- **Quản lý đồng thời (Concurrency)**: Backend sử dụng `asyncio.Semaphore(1)` kết hợp `asyncio.to_thread` để đảm bảo:
  1. Không chặn event loop của FastAPI (endpoint `GET /health` luôn phản hồi tức thì dưới 10ms).
  2. Các yêu cầu OCR nặng được xếp hàng thực thi tuần tự, ngăn ngừa hiện tượng nghẽn RAM và tràn CPU.

---

## 9. Chạy kiểm thử tự động (Unit & Integration Tests)

```powershell
# Chạy toàn bộ 104 Unit Tests (chạy nhanh < 2s)
python -m unittest discover -s tests -p "test_*.py" -v

# Chạy riêng bộ kiểm thử tích hợp API v2 trên model thật
python -m unittest tests/test_integration_v2.py -v
```
