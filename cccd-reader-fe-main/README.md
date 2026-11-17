# CCCD & GPLX READER FRONTEND (FLET DESKTOP)

Giao diện người dùng Desktop & Mobile viết bằng Flet (v1.1.2) tích hợp trực tiếp với backend FastAPI v2.

---

## 1. Tính năng nổi bật

- **Tự động phân loại hoặc chỉ định loại thẻ**: Hỗ trợ bộ chọn `Auto Detect`, `CCCD`, `GPLX`.
- **Lựa chọn chế độ xử lý**: Hỗ trợ `Fast Mode` (tối ưu tốc độ) và `Accurate Mode` (tối ưu độ chính xác qua Bounded Retry).
- **Trình bày dữ liệu dạng Form tương tác**:
  - Cho phép chỉnh sửa trực tiếp từng trường qua `TextField`.
  - Tự động gắn nhãn **`[Đã sửa]`** khi người dùng thay đổi giá trị so với OCR gốc.
  - Hiển thị icon trạng thái kiểm tra (Validation): Icon xanh lá khi hợp lệ, icon cam cảnh báo khi sai định dạng.
  - Hiển thị nhãn nguồn gốc trích xuất: `[Nắn phẳng]`, `[Crop YOLO]`, `[Chưa đọc được]`.
  - Loại bỏ hoàn toàn điểm tin cậy giả mạo (Zero Fake Confidence).
- **Sao chép JSON**: Nút sao chép toàn bộ dữ liệu cấu trúc vào Clipboard qua service `ft.Clipboard()`.
- **Xử lý sự cố tiếng Việt thân thiện**: Thông điệp lỗi chi tiết cho các mã lỗi 400, 413, 422, 503, Timeout và mất kết nối server.
- **An toàn Concurrency & Windows Desktop**:
  - Tự động vô hiệu hóa nút bấm khi đang xử lý để ngăn gửi request trùng lặp.
  - Nút "Thử lại" giúp gửi lại yêu cầu nhanh chóng khi mạng chập chờn.
  - Kiểm tra an toàn `can_use_camera` tránh phát sinh lỗi `FletUnsupportedPlatformException` trên Windows Desktop.

---

## 2. Hướng dẫn cài đặt & Khởi chạy

### Cài đặt thư viện:
```powershell
pip install flet==1.1.2 flet-cli==1.1.2 flet-desktop==1.1.2 httpx
```

### Chạy ở chế độ phát triển:
```powershell
flet run main.py
```

### Đóng gói thành file thực thi Windows (.exe):
```powershell
flet pack main.py -n "CCCD_Reader" -D --distpath "dist" -y
```
File thực thi độc lập sẽ được tạo tại `dist\CCCD_Reader\CCCD_Reader.exe`.
