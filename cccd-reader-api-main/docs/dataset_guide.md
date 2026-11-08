# HƯỚNG DẪN THU THẬP VÀ GÁN NHÃN DATASET YOLOV8 DOCUMENT DETECTION
## Dự án: Nhận dạng tài liệu CCCD & GPLX (XLA)

Tài liệu này hướng dẫn chi tiết quy trình thu thập dữ liệu hình ảnh và gán nhãn (annotation) chuẩn định dạng YOLO cho bài toán phát hiện hai loại giấy tờ tùy thân Việt Nam:
- **Lớp 0 (`cccd`)**: Căn cước công dân.
- **Lớp 1 (`gplx`)**: Giấy phép lái xe.

---

## 1. Cấu trúc thư mục Dataset chuẩn YOLO

Toàn bộ dữ liệu huấn luyện được đặt trong thư mục `cccd-reader-api-main/datasets/`:

```
cccd-reader-api-main/
├── configs/
│   └── data.yaml                  # File cấu hình đường dẫn và nhãn cho YOLO
└── datasets/
    ├── images/
    │   ├── train/                 # 70% tập ảnh huấn luyện
    │   ├── val/                   # 20% tập ảnh kiểm định trong lúc train
    │   └── test/                  # 10% tập ảnh đánh giá độc lập sau train
    └── labels/
        ├── train/                 # File nhãn .txt tương ứng với images/train
        ├── val/                   # File nhãn .txt tương ứng với images/val
        └── test/                  # File nhãn .txt tương ứng với images/test
```

> **Lưu ý**: Tên file ảnh và file nhãn phải trùng khớp hoàn toàn (chỉ khác phần mở rộng).  
> *Ví dụ:* `images/train/cccd_sample_001.jpg` $\leftrightarrow$ `labels/train/cccd_sample_001.txt`.

---

## 2. Tiêu chuẩn Thu thập Dữ liệu (Image Collection Guidelines)

Để mô hình YOLOv8n đạt độ khái quát hóa cao khi chụp từ camera điện thoại, quá trình thu thập ảnh cần tuân thủ các nguyên tắc sau:

### 2.1. Đa dạng môi trường & nền chụp (Background Variety)
- Đặt giấy tờ trên nhiều bề mặt khác nhau:
  - Bàn gỗ, bàn đá, mặt kính bàn làm việc.
  - Bàn màu trơn (trắng, đen, xanh, xám).
  - Nền có họa tiết (khăn trải bàn, vân gỗ phức tạp, thảm, sàn gạch men).
  - Bề mặt giấy tờ khác hoặc tài liệu văn phòng ngổn ngang.

### 2.2. Đa dạng góc chụp & Phối cảnh (Angle & Perspective)
- **Chụp trực diện**: Góc $90^\circ$ từ trên xuống (chiếm khoảng 30% tập dữ liệu).
- **Chụp nghiêng phối cảnh**: Góc nghiêng từ $15^\circ$ đến $45^\circ$ theo các hướng khác nhau (nghiêng trên-dưới, nghiêng trái-phải).
- **Chụp xoay góc**: Xoay thẻ các chiều $0^\circ, 90^\circ, 180^\circ, 270^\circ$ và xoay chéo tự do (đặc biệt phổ biến khi người dùng cầm điện thoại chụp tự do).

### 2.3. Điều kiện ánh sáng (Lighting Conditions)
- Ánh sáng ban ngày tự nhiên (trong phòng, gần cửa sổ).
- Ánh sáng đèn huỳnh quang, đèn LED văn phòng, đèn bàn vàng.
- Ảnh chụp có bóng đổ nhẹ (shadows) do tay hoặc điện thoại tạo ra.
- Ảnh có độ chói sáng nhẹ (glare/reflection từ màng bọc plastic thẻ PET).
- Ảnh trong điều kiện thiếu sáng nhẹ (low light).

### 2.4. Tỷ lệ kích thước thẻ trong khung hình (Scale & Distance)
- **Khoảng cách vừa phải**: Thẻ chiếm từ $40\% - 80\%$ diện tích khung hình.
- **Toàn cảnh xa hơn**: Thẻ chiếm khoảng $20\% - 30\%$ diện tích khung hình.
- **Cận cảnh**: Thẻ chiếm $80\% - 95\%$ khung hình.

### 2.5. Điều kiện tiên quyết của bài toán
- **Luôn đảm bảo 4 góc của thẻ nằm trọn vẹn trong khung hình** (không bị mép camera cắt mất góc).
- Bốn góc của thẻ không bị ngón tay, thẻ khác hoặc vật thể che khuất hoàn toàn (cho phép ngón tay chạm nhẹ ở mép nhưng 4 đỉnh góc phải quan sát được để phục vụ module Perspective Transform ở Giai đoạn sau).

---

## 3. Quy chuẩn Gán nhãn (Annotation Standard)

### 3.1. Danh mục lớp (Classes)
- `0`: **`cccd`** (Áp dụng cho CCCD gắn chip 2021, Căn cước 2024, CCCD mã vạch - cả mặt trước và mặt sau).
- `1`: **`gplx`** (Áp dụng cho Giấy phép lái xe mẫu thẻ PET mới, cả mặt trước và mặt sau).

### 3.2. Định dạng file nhãn YOLO (.txt)
Mỗi file `.txt` chứa một hoặc nhiều dòng, mỗi dòng đại diện cho một bounding box:
```
<class_id> <x_center> <y_center> <width> <height>
```
Trong đó:
- `class_id`: `0` hoặc `1`.
- `x_center`, `y_center`: Tọa độ tâm của bounding box, chuẩn hóa trong đoạn $[0.0, 1.0]$.
- `width`, `height`: Chiều rộng và chiều cao của bounding box, chuẩn hóa trong đoạn $[0.0, 1.0]$.

*Ví dụ:*
```
0 0.512340 0.489210 0.652100 0.421500
```

### 3.3. Quy tắc vẽ Bounding Box
- **Bao khít (Tight Bounding Box)**: Bounding box phải ôm sát các mép ngoài cùng của giấy tờ. Không vẽ quá rộng bao cả nền, không vẽ lẹm vào nội dung thẻ.
- **Trường hợp ảnh nghiêng/xoay (AABB - Axis-Aligned Bounding Box)**: Do định dạng YOLOv8 detection tiêu chuẩn sử dụng hộp chữ nhật song song với trục tọa độ ảnh, hộp chữ nhật phải bao trọn cả 4 góc của thẻ nghiêng (chứa trọn diện tích thẻ chiếu lên mặt phẳng ngang-dọc).
- **Trường hợp nhiều thẻ**: Nếu trong một ảnh có cả CCCD và GPLX hoặc 2 thẻ đặt cạnh nhau, hãy gán nhãn riêng cho từng thẻ trên các dòng khác nhau.

### 3.4. Công cụ gán nhãn đề xuất
1. **AnyLabeling / LabelStudio**:
   - Giao diện trực quan, cài đặt offline nhanh chóng bằng Python:
     ```bash
     pip install anylabeling
     anylabeling
     ```
   - Hỗ trợ xuất trực tiếp định dạng YOLO (`classes.txt` và các file nhãn `.txt`).
2. **LabelImg**:
   - Nhẹ, mã nguồn mở, hỗ trợ phím tắt chuyển đổi chế độ YOLO format (`W` để vẽ box, `D` để sang ảnh kế tiếp).
3. **Roboflow / CVAT**:
   - Dành cho gán nhãn trên nền web, hỗ trợ phân chia train/val tự động và augment ảnh trước khi xuất.

---

## 4. Bảo mật dữ liệu & Quyền riêng tư (Privacy Policy)

> [!CAUTION]
> **TUYỆT ĐỐI KHÔNG COMMIT HÌNH ẢNH GIẤY TỜ TÙY THÂN THẬT CÓ CHỨA THÔNG TIN CÁ NHÂN LÊN GITHUB HOẶC KHO CHỨA CÔNG KHAI!**

Các nguyên tắc an toàn dữ liệu bắt buộc:
1. File `.gitignore` của dự án đã được cấu hình tự động loại trừ thư mục `datasets/images/*` và `datasets/labels/*`.
2. Đối với việc phát triển và chạy thử trong môi trường chia sẻ:
   - Sử dụng thẻ mẫu mô hình (dummy card), thẻ demo đã được cấp phép.
   - Nếu dùng ảnh chụp thật để huấn luyện nội bộ, phải lưu trữ trên máy cá nhân cục bộ hoặc cloud riêng có mật khẩu bảo vệ.
   - Che mờ (blur/mask) các thông tin nhạy cảm: Số định danh cá nhân, số điện thoại, ảnh chân dung thực tế trước khi chia sẻ mẫu kiểm thử.

---

## 5. Kiểm tra dữ liệu trước khi Huấn luyện (Data Sanity Check)

Trước khi chạy script huấn luyện `train.py`, hãy thực hiện các bước kiểm tra sau:
1. Đảm bảo số lượng file `.jpg` trong `images/train` bằng đúng số file `.txt` trong `labels/train`.
2. Kiểm tra file `configs/data.yaml` trỏ đúng thư mục `datasets`.
3. Kiểm tra các nhãn `class_id` trong file txt chỉ nằm trong tập hợp `{0, 1}`.
4. Chạy script chuẩn bị và kiểm tra dataset:
   ```bash
   python -m scripts.train.dataset_prepare
   ```
