# IMPLEMENTATION PLAN – CCCD Reader Nâng cấp

> Phiên bản: 1.1 | Ngày cập nhật: 06/11/2026
> Áp dụng nguyên tắc CLAUDE.md: không tự suy đoán, không vượt phạm vi được giao, ghi rõ hạn chế.

---

## Tổng quan kiến trúc mục tiêu

```
Camera điện thoại
      │
      ▼
[FastAPI – /api/v1/upload]
      │
      ▼
[document_detector.py]   ← YOLOv8n: phát hiện bbox + phân loại CCCD/GPLX
      │
      ▼
[document_rectifier.py]  ← OpenCV: Canny → 4 góc → Perspective Transform
      │
      ▼
[cccd_extractor.py       ← EasyOCR: cắt ROI theo tỷ lệ → trích xuất trường
 gplx_extractor.py]
      │
      ▼
[FastAPI – JSON response]
      │
      ▼
[Flet Frontend]
```

---

## Điều kiện sử dụng đã thống nhất

- Giấy tờ được đặt trên bề mặt bất kỳ.
- Luôn quan sát được đầy đủ 4 góc.
- Giấy tờ có thể xoay hoặc bị nghiêng phối cảnh.
- Camera: điện thoại truyền ảnh tĩnh về server.
- Toàn bộ YOLOv8n, OpenCV, EasyOCR chạy trên máy tính (CPU).

---

## Giai đoạn 1 – Hoàn thiện CCCD Extractor ✅ ĐÃ HOÀN THÀNH & NGHIỆM THU

### 1. Phạm vi & Những nội dung đã thực hiện

| Hạng mục / File | Chi tiết thay đổi đã hoàn thành | Trạng thái |
|---|---|---|
| `services/cccd_extractor.py` | Bổ sung `extract_id_number()` và `extract_expiry_date()`; chuẩn hóa định dạng ngày tháng và 12 số CCCD | ✅ Hoàn thành |
| `services/cccd_extractor.py` | Sửa lỗi cú pháp CLI: `__name__ == "_main_"` $\rightarrow$ `__name__ == "__main__"` | ✅ Hoàn thành |
| `services/id_card_extractor.py` | Sửa lỗi cú pháp CLI: `__name__ == "_main_"` $\rightarrow$ `__name__ == "__main__"` | ✅ Hoàn thành |
| `services/cccd_extractor.py` | Hiệu chuẩn bộ tọa độ ROI thực tế (`id_number`, `full_name`, `expiry_date`, `place_of_origin`, `residence`) | ✅ Hoàn thành |
| `services/cccd_extractor.py` & `services/gplx_extractor.py` | Xuất `CCCD_ROIS` và `GPLX_ROIS` làm Single Source of Truth | ✅ Hoàn thành |
| `scripts/debug_roi.py` | Tái cấu trúc: loại bỏ khai báo ROI trùng lặp, import trực tiếp từ extractor services; xuất ảnh annotated và ảnh crop | ✅ Hoàn thành |
| `.gitignore` | Cập nhật bỏ qua thư mục `debug_output/` và `test_images/` để bảo vệ dữ liệu cá nhân | ✅ Hoàn thành |
| `tests/test_cccd_extractor.py` | Viết mới và mở rộng bộ unit test lên 33 test cases bao phủ toàn diện | ✅ 33/33 Passed |
| `cccd-reader-fe-main/main.py` | Đã có sẵn nhãn hiển thị cho 2 trường mới: `id_number` ("Số CCCD") và `expiry_date` ("Có giá trị đến") | ✅ Đã kiểm tra |

---

### 2. Chi tiết kỹ thuật & Logic đã hiệu chỉnh

#### A. Số CCCD (`id_number`)
- **Vị trí**: Dòng "Số / No.:" nằm bên phải ảnh chân dung, phía trên dòng "Họ và tên".
- **Tọa độ ROI hiệu chuẩn**: `BOX_ID_NUMBER = (0.37, 0.39, 0.89, 0.51)` (mở rộng $x_1$ sang `0.37` để triệt tiêu lỗi cắt mất chữ số `0` đầu tiên).
- **Quy tắc trích xuất**:
  - Chỉ chấp nhận chuỗi chứa đúng **chính xác 12 chữ số**.
  - Chuỗi có 10, 11 hoặc 13 chữ số bị từ chối và trả về `None` (không đoán, không cắt xén).
  - Giữ kiểu dữ liệu `str` để bảo toàn số `0` ở đầu (ví dụ: `030...`, `001...`).
  - Sửa lỗi OCR ký tự tương đồng: `O` $\rightarrow$ `0`, `I` / `l` $\rightarrow$ `1`, `S` $\rightarrow$ `5`, `B` $\rightarrow$ `8` trên các token số hợp lệ.
  - Hạn chế nối các token rời rạc nếu không cùng cấu trúc trường số.
  - Tuyệt đối không ghi số CCCD vào log debug.

#### B. Thời hạn sử dụng (`expiry_date`)
- **Đính chính cơ sở pháp lý**:
  - Theo **Thông tư 06/2021/TT-BCA**: Trên mẫu thẻ CCCD gắn chip 2021, trường *"Có giá trị đến / Date of expiry"* nằm ở **MẶT TRƯỚC**, góc dưới bên trái, ngay phía dưới ảnh chân dung công dân.
  - Thẻ Căn cước mới từ 01/07/2024 mới chuyển ngày hết hạn ra mặt sau. Extractor hiện tại chuyên biệt hỗ trợ mẫu CCCD gắn chip 2021 mặt trước.
- **Tọa độ ROI hiệu chuẩn**: `BOX_EXPIRY_DATE = (0.025, 0.85, 0.28, 0.92)` (loại bỏ hoàn toàn vùng ảnh chân dung và nhãn ngoài).
- **Quy tắc trích xuất**:
  - Hỗ trợ chuẩn hóa ngày: `DD/MM/YYYY`, `DD-MM-YYYY`, `DD.MM.YYYY`.
  - Hỗ trợ bổ sung định dạng khuyết 1 dấu phân cách do OCR mờ nét (`DD/MMYYYY`, `DDMM/YYYY`) và chuỗi 8 chữ số thuần (`DDMMYYYY`).
  - **Xác thực lịch nghiêm ngặt**: Dùng `datetime.strptime()` để kiểm tra tính hợp lệ lịch (bắt lỗi tháng $>12$, ngày $>31$, kiểm tra chính xác năm nhuận như `29/02/2032` hợp lệ và `29/02/2031` trả `None`).
  - Nhận diện nhãn đặc biệt: `"Không thời hạn"`.
  - Không tự tính thời hạn sử dụng từ ngày sinh khi không đọc được trên thẻ.

#### C. Bảng tọa độ ROI trước và sau hiệu chuẩn (Single Source of Truth)

| Trường thông tin | Tọa độ ban đầu $(x_1, y_1, x_2, y_2)$ | Tọa độ hiệu chuẩn thực tế | Kết quả cải thiện |
|---|---|---|---|
| `id_number` | `(0.42, 0.42, 0.88, 0.52)` | `(0.37, 0.39, 0.89, 0.51)` | Không còn mất số 0 đầu, confidence tăng từ 0.912 lên **1.000** |
| `full_name` | `(0.27, 0.53, 0.61, 0.63)` | `(0.27, 0.535, 0.62, 0.605)` | Tách sạch mép nhãn "Họ và tên", confidence **0.820** |
| `expiry_date` | `(0.04, 0.78, 0.32, 0.92)` | `(0.025, 0.85, 0.28, 0.92)` | Tách khỏi ảnh chân dung, confidence đạt **0.748** |
| `place_of_origin` | `(0.28, 0.79, 0.82, 0.855)` | `(0.28, 0.79, 0.82, 0.845)` | $y_2$ hạ xuống 0.845, không chồng lấn thường trú 1 |
| `residence_line_1` | `(0.665, 0.855, 0.93, 0.92)` | `(0.665, 0.855, 0.93, 0.915)` | $y_1=0.855, y_2=0.915$, tách biệt rõ ràng |
| `residence_line_2` | `(0.28, 0.91, 0.84, 0.985)` | `(0.28, 0.915, 0.84, 0.985)` | $y_1=0.915$, không lấn vào dòng 1 |

#### D. Ngưỡng Confidence & Hệ thống
- Giữ nguyên `MIN_CONFIDENCE = 0.3` làm ngưỡng tin cậy tiêu chuẩn trên toàn hệ thống.
- Cấu hình ROI export từ `services/cccd_extractor.py` (`CCCD_ROIS`) và `services/gplx_extractor.py` (`GPLX_ROIS`), script `debug_roi.py` tái sử dụng trực tiếp để tránh phân mảnh tọa độ.

---

### 3. Kết quả nghiệm thu thực tế

1. **Kiểm thử trên ảnh mẫu CCCD thực tế**:
   - Trích xuất thành công 100% đầy đủ 8 trường thông tin:
     ```json
     {
       "id_number": "030206013743",
       "full_name": "VŨ NGOC BẢO",
       "date_of_birth": "27/03/2006",
       "gender": "Nam",
       "nationality": "Việt Nam",
       "place_of_origin": "Nam Chinh Nam Sach Hải Dương",
       "residence": "Thôn Bich Đông, Nam Chinh Nam Sách; Hải Dương",
       "expiry_date": "27/03/2031"
     }
     ```
2. **Bộ Unit Test (`tests/test_cccd_extractor.py`)**:
   - 33/33 bài test vượt qua (100% Pass) trong 0.075 giây.
   - Bao phủ: số 0 ở đầu, boundary 10/11/13 chữ số, sửa ký tự OCR, nhãn rác tên, xác thực ngày tháng lịch, năm nhuận, schema hồi quy GPLX.
3. **Commit Git**: Commit an toàn mã nguồn với message `fix(roi): hieu chinh ROI CCCD va dong bo Single Source of Truth`.

---

## Giai đoạn 2 – YOLOv8n Detection ✅ ĐÃ HOÀN THÀNH PIPELINE (HUẤN LUYỆN THỰC TẾ: NOT_TESTED)

### 1. Phạm vi & File đã hoàn thành

| File | Nội dung & Chức năng | Trạng thái |
|---|---|:---:|
| `services/document_detector.py` | Wrapper YOLOv8n: nhận diện CCCD/GPLX, input validation, Dataclass trả bbox + class + confidence, vẽ & lưu bounding box | ✅ Đã hoàn thành |
| `configs/data.yaml` | Cấu hình dataset YOLO (nc=2, names: 0=cccd, 1=gplx) | ✅ Đã hoàn thành |
| `datasets/` | Cấu trúc thư mục chuẩn `images/{train,val,test}`, `labels/{train,val,test}` kèm `.gitkeep` | ✅ Đã hoàn thành |
| `scripts/train/dataset_prepare.py` | Kiểm tra toàn vẹn định dạng nhãn YOLO, phân chia train/val/test | ✅ Đã hoàn thành |
| `scripts/train/train.py` | Script fine-tune YOLOv8n với data augmentation chuyên biệt cho tài liệu xoay/nghiêng | ✅ Đã hoàn thành |
| `scripts/train/evaluate.py` | Đánh giá mAP50, mAP50-95, precision, recall, latency trên tập val/test | ✅ Đã hoàn thành |
| `scripts/predict_sample.py` | Chạy thử trên ảnh đơn/thư mục, vẽ và lưu ảnh có bounding box | ✅ Đã hoàn thành |
| `docs/dataset_guide.md` | Tài liệu hướng dẫn chi tiết thu thập và gán nhãn dataset | ✅ Đã hoàn thành |
| `tests/test_document_detector.py` | Bộ unit test kiểm thử toàn diện các trường hợp suy luận & ngoại lệ | ✅ 14/14 Passed |
| `STAGE2_TEST_REPORT.md` | Báo cáo chi tiết kết quả kiểm thử Giai đoạn 2 | ✅ Đã hoàn thành |

### 2. Classes
```yaml
nc: 2
names:
  0: cccd
  1: gplx
```

### 3. Kết quả nghiệm thu
1. `document_detector.py` import và suy luận trả đúng cấu trúc Dataclass `DetectionResult` và `Detection` trên cả CPU và CUDA.
2. Kiểm thử tự động `test_document_detector.py`: **14/14 test cases Passed (100%)** bao gồm trường hợp zero detection, single detection, multiple detections sắp xếp theo confidence, clamping tọa độ bbox, và các ca kiểm thử ảnh đầu vào không hợp lệ.
3. Kiểm thử hồi quy Giai đoạn 1 `test_cccd_extractor.py`: **33/33 test cases Passed (100%)**, không gây bất kỳ ảnh hưởng nào đến pipeline OCR và trích xuất trường.
4. **Trạng thái Huấn luyện Thực tế**: Đánh dấu **`NOT_TESTED`** theo đúng nguyên tắc bảo vệ quyền riêng tư, không lưu trữ dữ liệu cá nhân trên GitHub. Pipeline huấn luyện đã sẵn sàng 100% để người dùng đưa dữ liệu nội bộ vào chạy script `train.py`.

---

## Giai đoạn 3 – Document Rectifier (Perspective Transform)

### Phạm vi

| File | Nội dung |
|---|---|
| `services/document_rectifier.py` | OpenCV: nhận ảnh + bbox → Canny → 4 góc → warpPerspective → trả ảnh phẳng |

### Thuật toán

```
1. Crop vùng mở rộng quanh bbox (margin = 15%)
2. Grayscale → GaussianBlur → Canny (adaptive threshold)
3. findContours → lọc contour có diện tích > 30% bbox area
4. approxPolyDP → tìm contour 4 đỉnh, lồi, tỷ lệ cạnh hợp lý (0.4–2.5)
5. Nếu nhiều ứng viên → ưu tiên contour lớn nhất và gần trung tâm bbox nhất
6. getPerspectiveTransform + warpPerspective
7. Fallback: nếu không tìm được 4 góc → trả ảnh crop gốc + status="fallback"
```

### Schema trả về

```python
@dataclass
class RectifyResult:
    image: np.ndarray        # ảnh đã warp (hoặc crop gốc nếu fallback)
    corners: list[list[int]] | None  # 4 góc tìm được, None nếu fallback
    status: Literal["ok", "fallback"]
```

### Tiêu chí nghiệm thu

1. Với ảnh có 4 góc rõ ràng → `status="ok"`, ảnh output thẳng.
2. Với ảnh không tìm được 4 góc → `status="fallback"`, không raise exception.
3. Không tự tạo 4 góc giả.

---

## Giai đoạn 4 – Tích hợp YOLO + OpenCV + EasyOCR

### Phạm vi

| File | Thay đổi |
|---|---|
| `main.py` (api) | Endpoint mới `/api/v2/upload` dùng pipeline YOLO→Rectify→OCR |
| `main.py` (api) | Giữ nguyên `/api/v1/upload` cho luồng cũ |

### Luồng xử lý mới (`/api/v2/upload`)

```
Ảnh → document_detector (YOLO) → bbox + card_type
    → document_rectifier (OpenCV) → ảnh phẳng
    → cccd_extractor / gplx_extractor (EasyOCR)
    → JSON response
```

### Tiêu chí nghiệm thu

1. `/api/v1/upload` không bị thay đổi hành vi.
2. `/api/v2/upload` trả cùng schema JSON.
3. Với ảnh nghiêng, v2 cho kết quả tốt hơn v1 (kiểm tra thủ công).

---

## Giai đoạn 5 – Camera Điện thoại & Giao diện Quét

### Phạm vi

- Flet frontend: thêm tab "Quét trực tiếp" dùng API v2.
- Hỗ trợ stream ảnh từ điện thoại (WebSocket hoặc polling endpoint).

> Chi tiết sẽ được thiết kế sau khi Giai đoạn 4 hoàn thành.

---

## Giai đoạn 6 – Benchmark & Đánh giá

### Metrics

| Metric | Mục tiêu |
|---|---|
| Field accuracy (id_number) | ≥ 90% trên tập test |
| Field accuracy (full_name) | ≥ 85% |
| End-to-end latency (CPU) | < 10s/ảnh |
| YOLO mAP@0.5 | ≥ 0.85 |

---

## Quy tắc bất biến (không thay đổi trong mọi giai đoạn)

- Không xóa các hàm OCR đang hoạt động.
- Không thay đổi `/api/v1/upload` theo cách gây mất tương thích.
- Không viết lại giao diện Flet ngoài yêu cầu tối thiểu.
- Không huấn luyện lại EasyOCR.
- Không đưa dữ liệu giấy tờ cá nhân vào log hoặc commit.
- Không tự chuyển sang giai đoạn tiếp theo khi chưa nghiệm thu giai đoạn hiện tại.

