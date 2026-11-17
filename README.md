# CCCD & GPLX READER — TÀI LIỆU HỆ THỐNG VÀ HƯỚNG DẪN VẬN HÀNH (GIAI ĐOẠN 6)

Ứng dụng nhận diện, nắn chỉnh phối cảnh và trích xuất thông tin Căn cước công dân (CCCD) và Giấy phép lái xe (GPLX) Việt Nam sử dụng **YOLOv8n**, **OpenCV Document Rectification**, **EasyOCR** với bộ **Calibrated ROIs**, **FastAPI v2 Backend** và **Flet Desktop Frontend**.

---

## 1. Kiến trúc tổng thể hệ thống

Hệ thống được thiết kế theo kiến trúc **Client – Server tách rời (Decoupled Architecture)** tối ưu cho môi trường Windows Desktop:

```text
                  +----------------------------------------------+
                  |         FLET DESKTOP CLIENT (WINDOWS)        |
                  |     (Standalone Executable: CCCD_Reader.exe)  |
                  +----------------------------------------------+
                                         |
                            HTTP POST /api/v2/upload
                            (Timeout: 180s, Form-Data)
                                         v
                  +----------------------------------------------+
                  |              FASTAPI V2 BACKEND              |
                  |        (Uvicorn Single-Worker Process)       |
                  +----------------------------------------------+
                                         |
                         InferencePool (max_workers=1)
                       [Bảo vệ Concurrency & Chống OOM]
                                         |
               +-------------------------+-------------------------+
               |                                                   |
               v                                                   v
   +-----------------------+                           +-----------------------+
   |   YOLOv8n DETECTOR    |                           |  OPENCV RECTIFICATION  |
   | (Phát hiện & Phân loại|                           | (Canny, Contour, Hull |
   |  CCCD / GPLX 640x640) |                           |  WarpPerspective 856) |
   +-----------------------+                           +-----------------------+
               |                                                   |
               +-------------------------+-------------------------+
                                         |
                                         v
                        +---------------------------------+
                        |  DOCUMENT ORIENTATION ALIGNMENT |
                        | (Kiểm tra góc xoay 0/90/180/270)|
                        +---------------------------------+
                                         |
                                         v
                        +---------------------------------+
                        |    CALIBRATED ROI EXTRACTION    |
                        |   (Hiệu chuẩn theo chuẩn 856x540) |
                        +---------------------------------+
                                         |
                                         v
                        +---------------------------------+
                        |   ADAPTIVE OCR & VALIDATION     |
                        | (EasyOCR + Field Fallback/Retry)|
                        +---------------------------------+
                                         |
                                         v
                               Structured JSON Output
```

---

## 2. Yêu cầu hệ thống & Danh sách Dependencies

### 2.1. Yêu cầu phần cứng & Hệ điều hành
- **Hệ điều hành**: Windows 10/11 (64-bit).
- **CPU**: Tối thiểu 4 nhân (khuyên dùng Intel Core i5 / AMD Ryzen 5 trở lên).
- **RAM**: Tối thiểu 8 GB RAM.
- **Dung lượng đĩa trống**: Tối thiểu 5 GB (bao gồm mô hình PyTorch, EasyOCR weights và môi trường ảo).

### 2.2. Phiên bản nền tảng
- **Python**: `3.12.x` (64-bit).
- **Flet**: `1.1.2` (kèm `flet-cli 1.1.2`, `flet-desktop 1.1.2`).
- **FastAPI**: `0.162.0` (Uvicorn `0.54.0`).
- **PyTorch (CPU)**: `2.6.0+cpu` (Torchvision `0.21.0+cpu`).
- **Ultralytics**: `8.4.208` (YOLOv8n fine-tuned).
- **EasyOCR**: `1.7.2`.
- **OpenCV**: `opencv-python-headless 4.10.x`.

---

## 3. Cài đặt môi trường từ mã nguồn

### Bước 1: Khởi tạo môi trường ảo Python
Mở PowerShell tại thư mục gốc `C:\Users\bui nhu duc\XLA`:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Bước 2: Cài đặt Backend Dependencies
```powershell
pip install -r cccd-reader-api-main\requirements.txt
```

### Bước 3: Cài đặt Frontend & Build Dependencies
```powershell
pip install flet==1.1.2 flet-cli==1.1.2 flet-desktop==1.1.2 pyinstaller httpx
```

---

## 4. Hướng dẫn khởi chạy hệ thống

### 4.1. Khởi chạy Backend FastAPI (Terminal 1)
Khởi chạy Uvicorn server với **1 worker** (bắt buộc để đảm bảo Concurrency Pool kiểm soát an toàn tài nguyên CPU/RAM):
```powershell
cd "C:\Users\bui nhu duc\XLA\cccd-reader-api-main"
$env:PYTHONPATH="C:\Users\bui nhu duc\XLA\cccd-reader-api-main"
& "..\.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
```
- Endpoint API Swagger UI: `http://127.0.0.1:8000/docs`
- Endpoint Kiểm tra sức khỏe: `http://127.0.0.1:8000/health`

### 4.2. Khởi chạy Frontend Flet Desktop (Terminal 2)
```powershell
cd "C:\Users\bui nhu duc\XLA\cccd-reader-fe-main"
& "..\.venv\Scripts\flet.exe" run main.py
```

---

## 5. Hướng dẫn đóng gói ứng dụng Windows Desktop (`.exe`)

Ứng dụng Client được đóng gói độc lập bằng công cụ `flet pack` (sử dụng PyInstaller Engine tối ưu hóa riêng cho Flet/Flutter):

```powershell
cd "C:\Users\bui nhu duc\XLA"
& ".\.venv\Scripts\flet.exe" pack "cccd-reader-fe-main/main.py" `
    -n "CCCD_Reader" `
    -D `
    --distpath "dist" `
    -y
```

### Kết quả đóng gói:
- **Thư mục đầu ra**: `dist\CCCD_Reader\`
- **File thực thi**: `dist\CCCD_Reader\CCCD_Reader.exe` (Dung lượng: ~11.3 MB)
- **Tổng dung lượng thư mục**: ~120 MB (đã nhúng đầy đủ Python 3.12 runtime, Flutter rendering engine, OpenSSL và Visual C++ DLLs).
- **Tính độc lập**: Có thể copy toàn bộ thư mục `dist\CCCD_Reader` sang máy tính Windows 11 khác **không cần cài đặt Python** để sử dụng.

---

## 6. Cấu hình biến môi trường (Configuration)

| Tên biến | Mặc định | Mô tả |
|:---|:---:|:---|
| `CCCD_API_URL` | `http://127.0.0.1:8000/api/v2/upload` | Địa chỉ API Endpoint kết nối từ Frontend |
| `CCCD_API_TIMEOUT` | `180.0` | Thời gian chờ tối đa (giây) của request OCR |
| `MAX_CONCURRENT_INFERENCE` | `1` | Số lượng tác vụ OCR tối đa chạy đồng thời trên CPU backend |
| `DISABLE_MODEL_PRELOAD` | `0` (Không đặt) | Đặt `1` để bỏ qua nạp trước trọng số khi khởi động |

---

## 7. Quy trình bảo vệ an toàn dữ liệu & Quyền riêng tư (Privacy & Security)

Tuân thủ nghiêm ngặt các quy định về an toàn dữ liệu thông tin định danh cá nhân (PII):
1. **Không lưu trữ ảnh tạm**: Backend API v2 giải mã ảnh trực tiếp từ bộ nhớ RAM (`io.BytesIO` qua `cv2.imdecode`), hoàn toàn không ghi file ảnh CCCD/GPLX của người dùng ra ổ đĩa.
2. **Không ghi log thông tin cá nhân**: Toàn bộ hệ thống logging, debug log và unit test logs đều sử dụng cơ chế mặt nạ (Masking) `030******743` hoặc `[MASKED]`, tuyệt đối không ghi số định danh, họ tên, địa chỉ vào file log.
3. **Quản trị Clipboard**: Nút "Sao chép JSON" trên giao diện có thông báo SnackBar cảnh báo người dùng về việc dữ liệu nhạy cảm vừa được đưa vào Clipboard.
4. **Tự động dọn dẹp (Housekeeping)**: Endpoint v1 cũ tự động xóa file upload trong khối `finally: file_path.unlink()`. Đồng thời, sự kiện `lifespan` của FastAPI tự động quét và xóa sạch các file sót lại trong thư mục `uploads/` khi khởi động.

---

## 8. Troubleshooting & Khắc phục sự cố

| Hiện tượng | Nguyên nhân | Biện pháp khắc phục |
|:---|:---|:---|
| Lỗi `PermissionError: Access is denied: ...\.EasyOCR\model` | EasyOCR mặc định cố gắng ghi file vào thư mục người dùng khi chạy sandbox | Đặt biến môi trường hoặc chạy với quyền truy cập thư mục cache model đầy đủ |
| Lỗi `ValueError: binary mode doesn't take an encoding argument` khi chạy `flet pack` | Lỗi cú pháp trong `flet_cli/__pyinstaller/win_utils.py` dòng 149 của phiên bản 1.1.2 | Xóa tham số `encoding="utf-8"` trong hàm mở file nhị phân `open(exe_path, "ab")` |
| Client nhận lỗi `HTTP 413: FILE_TOO_LARGE` | Ảnh tải lên vượt quá 15MB | Sử dụng ảnh chụp chuẩn hoặc nén dung lượng file dưới 15MB trước khi upload |
| Client nhận lỗi `HTTP 422: NO_DOCUMENT_DETECTED` | Ảnh chụp quá mờ, bị che khuất góc thẻ hoặc không phải CCCD/GPLX | Đặt thẻ phẳng trên nền tương phản, chụp đủ sáng và lấy trọn 4 góc |
| Client nhận lỗi `HTTP 422: DOCUMENT_TYPE_MISMATCH` | Chọn chế độ thẻ CCCD nhưng tải lên GPLX (hoặc ngược lại) | Chuyển bộ chọn sang chế độ `Tự động phát hiện (Auto Detect)` |
| Giao diện báo `Không thể kết nối đến máy chủ` | Backend Uvicorn chưa được bật hoặc sai cổng | Kiểm tra cửa sổ Terminal chạy Backend đảm bảo đang hiển thị `Uvicorn running on http://127.0.0.1:8000` |
