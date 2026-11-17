# Vietnamese Document Pose Dataset (CCCD & GPLX 4-Corner Detection)

Bộ khung cấu trúc dữ liệu phục vụ huấn luyện mô hình **YOLOv8-Pose** cho bài toán phát hiện và định vị 4 góc vật lý thực tế của thẻ Căn cước công dân (CCCD) và Giấy phép lái xe (GPLX) Việt Nam.

---

## 1. Cấu trúc thư mục

```text
document_pose/
├── images/
│   ├── train/          # Ảnh tập huấn luyện
│   ├── val/            # Ảnh tập kiểm định (validation)
│   └── test/           # Ảnh tập kiểm thử độc lập (test)
├── labels/
│   ├── train/          # Nhãn YOLOv8-Pose tương ứng cho tập train
│   ├── val/            # Nhãn YOLOv8-Pose tương ứng cho tập val
│   └── test/           # Nhãn YOLOv8-Pose tương ứng cho tập test
├── card_pose.yaml      # Cấu hình dataset cho Ultralytics YOLOv8-Pose
├── dataset_manifest.json # Metadata quản lý trạng thái dataset
└── README.md           # Hướng dẫn chi tiết
```

---

## 2. Định dạng nhãn YOLOv8-Pose (Ultralytics Keypoint Format)

Mỗi file ảnh `<filename>.jpg` phải có một file nhãn tương ứng `<filename>.txt` trong thư mục `labels/`.
Mỗi dòng trong file nhãn đại diện cho một thẻ giấy tờ:

```text
<class_id> <x_center> <y_center> <width> <height> <x_tl> <y_tl> <v_tl> <x_tr> <y_tr> <v_tr> <x_br> <y_br> <v_br> <x_bl> <y_bl> <v_bl>
```

### 2.1. Quy ước Class ID
- `0`: `cccd` (Căn cước công dân gắn chip hoặc mã vạch)
- `1`: `gplx` (Giấy phép lái xe thẻ PET)

### 2.2. Bounding Box
- `x_center`, `y_center`, `width`, `height`: Tọa độ tâm và kích thước hộp bao quanh thẻ, được chuẩn hóa trong khoảng $[0.0, 1.0]$.

### 2.3. Quy ước 4 Keypoint (Thứ tự cố định)
4 điểm góc được định nghĩa theo hướng chuẩn của mặt thẻ (Upright frame of reference), theo chiều kim đồng hồ:
0. `(x_tl, y_tl, v_tl)`: **Top-Left** (Góc trên - trái)
1. `(x_tr, y_tr, v_tr)`: **Top-Right** (Góc trên - phải)
2. `(x_br, y_br, v_br)`: **Bottom-Right** (Góc dưới - phải)
3. `(x_bl, y_bl, v_bl)`: **Bottom-Left** (Góc dưới - trái)

Tất cả tọa độ $x, y$ đều được chuẩn hóa theo kích thước ảnh gốc $[0.0, 1.0]$.

### 2.4. Trạng thái Visibility Flag ($v$)
- `v = 2`: Góc nhìn thấy rõ ràng (Visible and clearly identified).
- `v = 1`: Góc bị che khuất một phần (bởi ngón tay, chói sáng) hoặc là **giao điểm ngoại suy của góc bo tròn 3.18mm** (Occluded / Extrapolated corner).
- `v = 0`: Góc hoàn toàn nằm ngoài khung hình ảnh hoặc không xác định được.

> [!IMPORTANT]
> **Quy tắc gán nhãn góc bo tròn 3.18mm:**
> Thẻ CCCD và GPLX theo chuẩn ISO/IEC 7810 ID-1 có 4 góc bo tròn bán kính $R = 3.18\text{ mm}$.  
> Khi gán nhãn, **phải đặt keypoint tại giao điểm ngoại suy của hai cạnh thẳng liền kề**, không chấm vào mép đường cong. Điều này bảo đảm khi nắn phối cảnh (Perspective Transform), ảnh kết quả giữ nguyên tỉ lệ chữ nhật $85.6 \times 53.98\text{ mm}$ hoàn hảo.

---

## 3. Hướng dẫn Export từ Roboflow hoặc CVAT

### Xuất từ Roboflow:
1. Tạo Project loại **Keypoint Detection** (Pose Estimation).
2. Thiết lập 4 keypoints với tên: `top_left`, `top_right`, `bottom_right`, `bottom_left`.
3. Gán nhãn các ảnh thẻ với đầy đủ Bounding Box và 4 góc.
4. Khi xuất dữ liệu: Chọn format **YOLOv8 Pose**.
5. Giải nén nội dung vào thư mục này:
   - Các file trong `train/images/` chuyển vào `images/train/`.
   - Các file trong `train/labels/` chuyển vào `labels/train/`.
   - Tương tự cho `val` và `test`.

---

## 4. Kiểm tra tính toàn vẹn của Dataset

Trước khi huấn luyện, luôn chạy script kiểm tra:

```powershell
python scripts/pose/validate_pose_dataset.py --data datasets/document_pose/card_pose.yaml
```

Script sẽ kiểm tra:
- Khớp số lượng giữa file ảnh và file nhãn (1-1 pairing).
- Tính hợp lệ của giá trị tọa độ $[0.0, 1.0]$ và tính lồi của 4 góc.
- Kiểm tra trùng lặp (Data leakage) giữa các split train/val/test bằng content hash.
- Thống kê tỷ lệ phân bổ class và visibility flags.

---

## 5. Quy tắc phân chia dữ liệu (Splits)

- **Train (70% - 80%):** Dùng để huấn luyện trọng số mô hình.
- **Val (10% - 15%):** Dùng để tuning siêu tham số và early stopping.
- **Test (10% - 15%):** Dùng để kiểm thử mù độc lập sau khi huấn luyện xong.
- **Lưu ý:** Tất cả ảnh của cùng một người hoặc cùng một thẻ (chụp ở các góc khác nhau) phải nằm chung trong một split duy nhất, không được chia sang split khác để tránh Data Leakage.

---

## 6. Chính sách bảo mật dữ liệu cá nhân (PII Policy)

- **TUYỆT ĐỐI KHÔNG** commit ảnh thẻ thật, ảnh chứa dữ liệu cá nhân (họ tên, số CMND/CCCD, địa chỉ, ảnh chân dung thực) lên GitHub repository công khai.
- Thư mục `images/` và `labels/` đã được cấu hình trong `.gitignore` để ngăn chặn vô tình commit dữ liệu nhạy cảm.
- Khi chia sẻ dataset nội bộ phục vụ huấn luyện, phải thực hiện qua kênh lưu trữ bảo mật được mã hóa.
