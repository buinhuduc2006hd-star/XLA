# IMPLEMENTATION PLAN – CCCD Reader Nâng cấp

> Phiên bản: 1.0 | Ngày tạo: 06/11/2026
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

## Giai đoạn 1 – Hoàn thiện CCCD Extractor ✅ ĐANG THỰC HIỆN

### Phạm vi

| File | Thay đổi |
|---|---|
| `services/cccd_extractor.py` | Thêm `extract_id_number()` và `extract_expiry_date()` |
| `cccd-reader-fe-main/main.py` | Thêm nhãn hiển thị cho 2 trường mới |
| `tests/test_cccd_extractor.py` | Unit test cho 2 hàm mới (không cần ảnh thật) |

### Hạn chế đã xác nhận

**Trường `expiry_date` ("Có giá trị đến"):**
- Trên **CCCD chip mới (2021+)**, trường này nằm ở **MẶT SAU**.
- Hệ thống hiện tại nhận 1 ảnh qua `/api/v1/upload` không phân biệt mặt trước/sau.
- Giải pháp Giai đoạn 1: thêm vùng ROI cho mặt sau (khu vực phía dưới), nếu không OCR được → trả `None` và log rõ lý do.
- Giải pháp dài hạn (Giai đoạn 4+): API nhận 2 ảnh riêng biệt (mặt trước + mặt sau).

**Tọa độ ROI `id_number`:**
- Ước tính theo layout chuẩn Bộ Công an 2021. Cần kiểm tra lại với ảnh CCCD thật.
- ROI được đánh dấu `# CALIBRATION NEEDED` trong code.

### Tiêu chí nghiệm thu

1. `extract_id_number()` trả 12 chữ số hoặc `None` – không bao giờ trả chuỗi < hoặc > 12 ký tự.
2. `extract_expiry_date()` trả định dạng `DD/MM/YYYY` hoặc `None`.
3. `extract_cccd_fields()` output chứa 2 key mới (`id_number`, `expiry_date`).
4. Unit test toàn bộ pass.
5. GPLX extractor không bị thay đổi (kiểm tra bằng import test).

---

## Giai đoạn 2 – YOLOv8n Detection

### Phạm vi

| File | Nội dung |
|---|---|
| `services/document_detector.py` | Wrapper YOLOv8n: load model, predict, trả bbox + class + confidence |
| `scripts/train/dataset_prepare.py` | Tạo cấu trúc dataset YOLO format (images/labels) |
| `scripts/train/train.py` | Script fine-tune YOLOv8n trên dataset CCCD/GPLX |
| `scripts/train/evaluate.py` | mAP, precision, recall trên tập val |
| `scripts/train/predict_sample.py` | Chạy thử trên ảnh đơn, vẽ bbox lên ảnh |
| `configs/dataset.yaml` | Cấu hình dataset YOLO (nc=2, names=[cccd, gplx]) |
| `docs/dataset_guide.md` | Hướng dẫn thu thập và gán nhãn ảnh |

### Classes

```yaml
nc: 2
names:
  0: cccd
  1: gplx
```

### Tiêu chí nghiệm thu

1. `document_detector.py` import được, predict trả đúng schema khi có model.
2. Script train chạy được với dataset mẫu nhỏ (≥ 10 ảnh/class).
3. Có hướng dẫn rõ ràng về cách thu thập và gán nhãn.

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

