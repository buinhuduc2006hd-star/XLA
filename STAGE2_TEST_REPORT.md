# BÁO CÁO KIỂM THỬ GIAI ĐOẠN 2 – YOLOV8N DOCUMENT DETECTION

> **Dự án**: XLA – Hệ thống Tự động Nhận dạng CCCD & GPLX  
> **Phiên bản**: 2.0 | **Ngày hoàn thành**: 09/11/2026  
> **Người thực hiện**: Antigravity Pair Programmer  
> **Trạng thái**: ✅ **HOÀN THÀNH PIPELINE & PASS TOÀN BỘ KIỂM THỬ KHẢ THI**

---

## 1. Mục tiêu Giai đoạn 2

Triển khai module **YOLOv8n Document Detection** phục vụ việc tự động phát hiện và khoanh vùng 2 loại giấy tờ tùy thân Việt Nam trên mọi bề mặt:
- Lớp 0: `cccd` (Căn cước công dân).
- Lớp 1: `gplx` (Giấy phép lái xe).

### Điều kiện và phạm vi ràng buộc:
- Bốn góc của giấy tờ luôn quan sát được đầy đủ trong khung hình.
- Chấp nhận ảnh chụp xoay các chiều ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) và nghiêng phối cảnh ($15^\circ - 45^\circ$).
- Chạy suy luận (inference) trên CPU trước, tự động chuyển sang CUDA nếu máy tính có GPU.
- **Không giả định** trọng số mặc định `yolov8n.pt` (huấn luyện trên COCO 80 lớp) đã nhận diện được CCCD/GPLX.
- **Không commit** dữ liệu giấy tờ cá nhân thật lên GitHub.
- **Không can thiệp** vào EasyOCR, FastAPI endpoint, Flet UI hay thuật toán Perspective Transform (giữ nguyên cho Giai đoạn tiếp theo).

---

## 2. Danh mục Module & File Đã Xây Dựng

| STT | File / Đường dẫn | Vai trò & Chức năng | Trạng thái |
|:---:|---|---|:---:|
| 1 | `cccd-reader-api-main/services/document_detector.py` | Module chính `DocumentDetector`: nhận diện, validate input, trả dataclass chuẩn, vẽ & lưu bounding box | ✅ Đã hoàn thành |
| 2 | `cccd-reader-api-main/configs/data.yaml` | File cấu hình dataset chuẩn YOLO (nc=2, names: 0=cccd, 1=gplx) | ✅ Đã hoàn thành |
| 3 | `cccd-reader-api-main/datasets/` | Cấu trúc thư mục chuẩn `images/{train,val,test}` và `labels/{train,val,test}` kèm `.gitkeep` | ✅ Đã hoàn thành |
| 4 | `cccd-reader-api-main/scripts/train/dataset_prepare.py` | Kiểm tra toàn vẹn dataset, validate định dạng nhãn txt, phân chia train/val/test | ✅ Đã hoàn thành |
| 5 | `cccd-reader-api-main/scripts/train/train.py` | Script fine-tune YOLOv8n với các tham số augmentation chuyên biệt cho thẻ xoay/nghiêng | ✅ Đã hoàn thành |
| 6 | `cccd-reader-api-main/scripts/train/evaluate.py` | Đánh giá mô hình trên split val/test (mAP50, mAP50-95, precision, recall, latency) | ✅ Đã hoàn thành |
| 7 | `cccd-reader-api-main/scripts/predict_sample.py` | Script dự đoán ảnh đơn/thư mục, vẽ bounding box trực quan và lưu ảnh kết quả | ✅ Đã hoàn thành |
| 8 | `cccd-reader-api-main/docs/dataset_guide.md` | Hướng dẫn chi tiết quy trình thu thập ảnh, đa dạng hóa nền/góc chụp và quy chuẩn gán nhãn YOLO | ✅ Đã hoàn thành |
| 9 | `cccd-reader-api-main/tests/test_document_detector.py` | Bộ unit test kiểm thử: zero detection, single detection, multiple detections, invalid inputs, clamping, annotation | ✅ 14/14 Passed |
| 10 | `.gitignore` | Cấu hình bảo vệ dữ liệu: loại trừ `bao/`, `datasets/images/*`, `datasets/labels/*`, `runs/`, `*.pt` | ✅ Đã cập nhật |

---

## 3. Thiết kế Kỹ thuật Chi tiết

### 3.1. Chuẩn hóa Đầu vào & Đầu ra (`document_detector.py`)
- **Đầu vào (Input)**:
  - Hỗ trợ cả đường dẫn `str`, `pathlib.Path` và mảng NumPy BGR `np.ndarray`.
  - Bộ kiểm tra `_validate_image_input()` chặn sớm các lỗi: file không tồn tại (`FileNotFoundError`), sai kiểu dữ liệu (`TypeError`), mảng rỗng hoặc sai số chiều/channel (`ValueError`), kích thước ảnh $< 10\times 10$ (`ValueError`).
- **Đầu ra (Output)**:
  - Định nghĩa 2 Dataclass rõ ràng: `Detection` và `DetectionResult`.
  - Trả về: `class_id`, `class_name` ("cccd" / "gplx"), `confidence` (float), `bbox_xyxy` (4 số nguyên $x_1, y_1, x_2, y_2$ đã kẹp an toàn trong kích thước ảnh).
  - Tự động sắp xếp các detection theo `confidence` giảm dần; cung cấp thuộc tính `top_detection` và `has_detection`.

### 3.2. Trực quan hóa Bounding Box (`annotate_image`)
- Tự động tính toán độ dày nét vẽ (`line_thickness`) và kích thước chữ dựa trên độ phân giải ảnh.
- Phối màu trực quan theo lớp:
  - CCCD: Màu Xanh lá (`Sea Green` - BGR `(46, 139, 87)`).
  - GPLX: Màu Xanh lam / Cam vàng (`Steel Blue` - BGR `(218, 112, 37)`).
- Vẽ khung nền cho nhãn text để đảm bảo độ tương phản cao, dễ quan sát trên mọi màu nền ảnh.
- Hỗ trợ lưu ra file theo tham số `save_path`.

### 3.3. Pipeline Huấn luyện & Data Augmentation (`train.py`)
- Cấu hình siêu tham số fine-tune tối ưu cho giấy tờ phẳng:
  - `imgsz=640`, `batch=8` hoặc `16`.
  - `degrees=15.0`: Xoay ngẫu nhiên $\pm 15^\circ$ mô phỏng việc đặt thẻ lệch góc.
  - `perspective=0.0005`: Biến dạng phối cảnh nhẹ mô phỏng camera điện thoại chụp nghiêng.
  - `scale=0.2`: Thu phóng tỷ lệ thẻ trong khung hình.
  - `fliplr=0.5`: Lật gương ngang.
  - `flipud=0.0`: Không lật ngược theo chiều dọc để bảo toàn chiều chữ.

---

## 4. Kết quả Kiểm thử Toàn diện

### 4.1. Unit Test Giai đoạn 2 (`test_document_detector.py`)
Chạy kiểm thử với lệnh:
```bash
.venv\Scripts\python.exe -m unittest cccd-reader-api-main/tests/test_document_detector.py -v
```

Kết quả: **14/14 tests PASSED (100%)** trong `0.082s`.

| Nhóm kiểm thử | Tên Test Case | Mục đích & Tiêu chí nghiệm thu | Kết quả |
|---|---|---|:---:|
| **Xử lý ngoại lệ đầu vào (Input Validation)** | `test_non_existent_file_raises_filenotfound` | Đường dẫn file không tồn tại $\rightarrow$ raise `FileNotFoundError` | ✅ Pass |
| | `test_unsupported_type_raises_typeerror` | Input là kiểu `int`, `list` $\rightarrow$ raise `TypeError` | ✅ Pass |
| | `test_empty_numpy_array_raises_valueerror` | Mảng NumPy rỗng $\rightarrow$ raise `ValueError` | ✅ Pass |
| | `test_wrong_dimensions_raises_valueerror` | Mảng 1D, 2D (grayscale), 4D $\rightarrow$ raise `ValueError` | ✅ Pass |
| | `test_too_small_dimensions_raises_valueerror` | Ảnh nhỏ hơn $10\times 10$ $\rightarrow$ raise `ValueError` | ✅ Pass |
| | `test_corrupt_file_raises_valueerror` | File rác không giải mã được ảnh $\rightarrow$ raise `ValueError` | ✅ Pass |
| **Logic Suy luận (Inference Cases)** | `test_case_no_detection` | Không phát hiện được thẻ $\rightarrow$ `has_detection=False`, `top_detection=None` | ✅ Pass |
| | `test_case_single_detection_cccd` | Phát hiện 1 thẻ CCCD $\rightarrow$ đúng `class_id=0`, `class_name='cccd'`, conf và bbox | ✅ Pass |
| | `test_case_single_detection_gplx` | Phát hiện 1 thẻ GPLX $\rightarrow$ đúng `class_id=1`, `class_name='gplx'`, conf và bbox | ✅ Pass |
| | `test_case_multiple_detections_sorting` | Phát hiện nhiều thẻ $\rightarrow$ tự động sắp xếp giảm dần theo confidence | ✅ Pass |
| | `test_bbox_coordinates_clamped_to_image_boundary` | Tọa độ bbox ngoài rìa ảnh $\rightarrow$ kẹp an toàn vào $[0, W-1]$ và $[0, H-1]$ | ✅ Pass |
| **Trực quan hóa (Annotation)** | `test_annotate_image_and_save` | Vẽ bounding box và lưu ảnh kết quả vào file thành công | ✅ Pass |
| **Toàn vẹn Cấu hình (Dataset Config)** | `test_data_yaml_exists_and_valid` | File `data.yaml` tồn tại, đủ 2 lớp `cccd` & `gplx` | ✅ Pass |
| | `test_dataset_prepare_checker` | Script `check_dataset_integrity` quét toàn vẹn thư mục dataset | ✅ Pass |

---

### 4.2. Kiểm thử Hồi quy Giai đoạn 1 (`test_cccd_extractor.py`)
Chạy kiểm thử với lệnh:
```bash
.venv\Scripts\python.exe -m unittest cccd-reader-api-main/tests/test_cccd_extractor.py -v
```

Kết quả: **33/33 tests PASSED (100%)** trong `0.074s`.  
Toàn bộ logic trích xuất 8 trường của CCCD, 7 trường của GPLX và tính toàn vẹn của ROI Single Source of Truth được giữ nguyên vẹn 100%, không xảy ra bất kỳ lỗi hồi quy nào.

---

## 5. Báo cáo Trạng thái Huấn luyện Thực tế (Training Status)

> [!NOTE]
> **Trạng thái Huấn luyện**: **`NOT_TESTED`** (Bảo lưu theo đúng yêu cầu đề bài).

- **Lý do**:
  1. Theo quy tắc bảo mật và quyền riêng tư, kho lưu trữ mã nguồn GitHub không được phép chứa dữ liệu giấy tờ thật của cá nhân.
  2. Toàn bộ pipeline (cấu trúc thư mục dataset chuẩn YOLO, file `data.yaml`, script phân chia dữ liệu `dataset_prepare.py`, script train `train.py`, script evaluate `evaluate.py`, script predict `predict_sample.py`) đã sẵn sàng 100%.
- **Cơ chế An toàn**:
  - Khi chưa có trọng số custom `best.pt`, `DocumentDetector` chủ động phát hiện và ghi log `WARNING`, nhắc nhở người dùng cần fine-tune trọng số riêng, không giả định mô hình gốc tự động phát hiện được thẻ.

---

## 6. Hướng dẫn Vận hành Huấn luyện Thực tế Khi Đưa Dữ liệu Vào

Khi người dùng đã chuẩn bị dữ liệu ảnh CCCD/GPLX (theo đúng hướng dẫn tại `cccd-reader-api-main/docs/dataset_guide.md`):

1. **Kiểm tra dữ liệu trước khi train**:
   ```bash
   python -m scripts.train.dataset_prepare --check
   ```
2. **Khởi chạy Fine-tune YOLOv8n**:
   ```bash
   python -m scripts.train.train --epochs 50 --batch 8 --imgsz 640
   ```
   *(Mô hình tốt nhất sau khi train sẽ tự động được lưu tại `runs/detect/train_document/weights/best.pt` và copy vào `models/yolov8n_document.pt`)*.
3. **Đánh giá mAP trên tập Test**:
   ```bash
   python -m scripts.train.evaluate --weights models/yolov8n_document.pt --split test
   ```
4. **Dự đoán và hiển thị kết quả trên ảnh mẫu**:
   ```bash
   python -m scripts.predict_sample path/to/image.jpg --weights models/yolov8n_document.pt --save-dir runs/predict
   ```

---

## 7. Kết luận & Sẵn sàng cho Giai đoạn tiếp theo

Giai đoạn 2 đã được triển khai hoàn chỉnh, đầy đủ mã nguồn, tài liệu hướng dẫn và bộ unit test tự động. Tất cả các ràng buộc về an toàn dữ liệu, tính tương thích ngược và yêu cầu kỹ thuật đều được đáp ứng tuyệt đối.

Hệ thống đã sẵn sàng để bước sang **Giai đoạn 3: Document Rectifier (OpenCV Canny $\rightarrow$ 4 góc $\rightarrow$ Perspective Transform)**.

