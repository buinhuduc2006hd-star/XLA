# TEST_REPORT.md – Báo cáo Kiểm thử và Hiệu chỉnh Giai đoạn 1

> Dự án: CCCD Reader (XLA)  
> Commit hash sau khi sửa: `d3b2412e74dfc89d1136b90880da2dada4983467`  
> Người thực hiện: Antigravity  
> Thời gian: 06/11/2026  

---

## 1. Danh sách file đã sửa và tạo mới

| File | Hành động | Mục đích thay đổi |
|---|---|---|
| `cccd-reader-api-main/services/cccd_extractor.py` | Sửa | • Định vị lại trường Thời hạn sử dụng về mặt trước (dưới ảnh chân dung theo Thông tư 06/2021/TT-BCA).<br>• Siết chặt logic nhận diện Số CCCD (chính xác 12 chữ số, không cắt/thêm).<br>• Ngăn chặn ghi log thông tin số CCCD.<br>• Sửa lỗi CLI `__name__ == "__main__"`. |
| `cccd-reader-api-main/services/id_card_extractor.py` | Sửa | • Sửa lỗi CLI entrypoint `if __name__ == "__main__":`. |
| `cccd-reader-api-main/services/gplx_extractor.py` | Sửa | • Sửa lỗi CLI entrypoint `if __name__ == "__main__":`. |
| `cccd-reader-api-main/scripts/debug_roi.py` | Tạo mới | • Công cụ vẽ tất cả ROI lên ảnh mẫu, xuất ảnh cắt từng trường, hiển thị bảng tọa độ relative/pixel. |
| `cccd-reader-api-main/tests/test_cccd_extractor.py` | Cập nhật | • Bộ 26 unit tests kiểm tra toàn bộ các ca thành công, biên, lỗi, và hồi quy schema. |

---

## 2. Những lỗi đã khắc phục

### 2.1. Vị trí trường Thời hạn sử dụng (Có giá trị đến)
- **Lỗi cũ:** Nhận định sai trường này nằm ở mặt sau CCCD chip 2021 (dẫn đến box nằm ở tọa độ mặt sau `0.50, 0.75, 0.95, 0.88`).
- **Khắc phục:** 
  - Cập nhật tài liệu kỹ thuật: Xác định rõ mẫu thẻ đang hỗ trợ là **CCCD gắn chip 2021 (Thông tư 06/2021/TT-BCA)**.
  - Cập nhật tọa độ ROI `BOX_EXPIRY_DATE: (0.04, 0.78, 0.32, 0.92)` nằm ở **mặt trước, góc dưới bên trái, ngay phía dưới ảnh chân dung**.
  - Không nhầm lẫn với mẫu Thẻ Căn cước mới (từ 01/07/2024 theo Luật Căn cước 2023).

### 2.2. Logic Số CCCD (`id_number`)
- **Lỗi cũ:** Dùng `_ID_12_PATTERN.search(digits_only)`, dẫn đến chuỗi 13 chữ số bị cắt lấy 12 số đầu và vẫn được chấp nhận là hợp lệ.
- **Khắc phục:**
  - Quy tắc nghiêm ngặt: Bắt buộc độ dài chuỗi chữ số phải **chính xác 12 chữ số** (`len == 12`).
  - Chuỗi 11 số hoặc 13 số đều trả về `None`, không đoán thêm và không cắt xén.
  - Bảo toàn định dạng chuỗi (`str`), không ép kiểu số nguyên để không mất chữ số `0` ở đầu (ví dụ: `001...`).
  - Chỉ áp dụng sửa ký tự nhầm (O/0, I/1, l/1, S/5, B/8...) khi chuỗi có độ dài tiềm năng (12 ký tự) và phần lớn là số. Không ép chuyển đổi các chuỗi chữ như họ tên tiếng Việt.
  - Kiểm tra từng đoạn/dòng OCR riêng biệt trước, tránh ghép các dòng khác trường tạo thành số 12 chữ số giả mạo.
  - **Bảo mật:** Không ghi số CCCD ra log debug.

### 2.3. Lỗi CLI Entrypoint
- **Lỗi cũ:** Cả 3 file `cccd_extractor.py`, `id_card_extractor.py`, `gplx_extractor.py` đều viết sai thành `if __name__ == "_main_":` (thiếu dấu gạch dưới).
- **Khắc phục:** Đã sửa thành `if __name__ == "__main__":` chuẩn Python, chạy được trực tiếp từ terminal.

### 2.4. Công cụ Hiệu chuẩn ROI
- **Đã tạo:** `scripts/debug_roi.py` cho phép:
  - Vẽ trực quan các ROI lên ảnh giấy tờ.
  - Xuất ảnh crop của từng trường vào thư mục `debug_output/crops/`.
  - Hỗ trợ cả 2 loại thẻ `cccd` và `gplx`.
  - Cho phép tùy biến override tọa độ qua JSON config mà không cần sửa code.

---

## 3. Kết quả kiểm thử Unit Test

Đã chạy kiểm thử với `python -m unittest tests.test_cccd_extractor -v`:

```text
test_cccd_full_schema (tests.test_cccd_extractor.TestCccdSchemaAndRegression.test_cccd_full_schema)
extract_cccd_fields trả đúng 8 trường theo đúng thứ tự. ... ok
test_gplx_regression_schema (tests.test_cccd_extractor.TestCccdSchemaAndRegression.test_gplx_regression_schema)
extract_gplx_fields giữ nguyên 7 trường, không dính trường của CCCD. ... ok
test_empty_returns_none (tests.test_cccd_extractor.TestExtractExpiryDate.test_empty_returns_none)
Vùng không đọc được -> None. ... ok
test_invalid_day_returns_none (tests.test_cccd_extractor.TestExtractExpiryDate.test_invalid_day_returns_none)
Ngày 32 không tồn tại -> None. ... ok
test_invalid_month_returns_none (tests.test_cccd_extractor.TestExtractExpiryDate.test_invalid_month_returns_none)
Tháng 13 không tồn tại -> None. ... ok
test_leap_year_valid (tests.test_cccd_extractor.TestExtractExpiryDate.test_leap_year_valid)
Năm nhuận 2032 có ngày 29/02 hợp lệ. ... ok
test_no_expiry_limit_label (tests.test_cccd_extractor.TestExtractExpiryDate.test_no_expiry_limit_label)
Trường hợp 'Không thời hạn'. ... ok
test_non_leap_year_invalid (tests.test_cccd_extractor.TestExtractExpiryDate.test_non_leap_year_invalid)
Năm không nhuận 2031 không có ngày 29/02 -> None. ... ok
test_random_text_without_date_returns_none (tests.test_cccd_extractor.TestExtractExpiryDate.test_random_text_without_date_returns_none)
Không chứa ngày tháng -> None. ... ok
test_valid_expiry_dot_and_hyphen (tests.test_cccd_extractor.TestExtractExpiryDate.test_valid_expiry_dot_and_hyphen)
Định dạng dấu chấm hoặc gạch ngang được chuẩn hóa thành /. ... ok
test_valid_expiry_embedded_in_label (tests.test_cccd_extractor.TestExtractExpiryDate.test_valid_expiry_embedded_in_label)
Ngày nằm cùng dòng với nhãn tiếng Việt/Anh. ... ok
test_valid_expiry_slash (tests.test_cccd_extractor.TestExtractExpiryDate.test_valid_expiry_slash)
Định dạng DD/MM/YYYY chuẩn. ... ok
test_eleven_digits_returns_none (tests.test_cccd_extractor.TestExtractIdNumber.test_eleven_digits_returns_none)
Thiếu 1 chữ số (11 số) -> Bắt buộc trả None, không đoán. ... ok
test_empty_ocr_returns_none (tests.test_cccd_extractor.TestExtractIdNumber.test_empty_ocr_returns_none)
Không có kết quả OCR -> trả None. ... ok
test_leading_zero_preserved (tests.test_cccd_extractor.TestExtractIdNumber.test_leading_zero_preserved)
Số CCCD bắt đầu bằng số 0 (ví dụ mã Hà Nội 001) phải giữ nguyên dạng chuỗi. ... ok
test_multiple_fields_confusion_prevents_false_combination (tests.test_cccd_extractor.TestExtractIdNumber.test_multiple_fields_confusion_prevents_false_combination)
Không ghép văn bản lộn xộn từ trường khác tạo thành 12 số. ... ok
test_multiple_parts_combining_valid_digits (tests.test_cccd_extractor.TestExtractIdNumber.test_multiple_parts_combining_valid_digits)
Các cụm số bị tách thành nhiều token nhưng tổng đúng 12 chữ số. ... ok
test_ocr_fix_I_and_l_to_1 (tests.test_cccd_extractor.TestExtractIdNumber.test_ocr_fix_I_and_l_to_1)
Sửa lỗi chữ I hoa hoặc l thường thành số 1. ... ok
test_ocr_fix_O_to_0 (tests.test_cccd_extractor.TestExtractIdNumber.test_ocr_fix_O_to_0)
Sửa lỗi chữ O hoa thành số 0. ... ok
test_ocr_fix_S_and_B (tests.test_cccd_extractor.TestExtractIdNumber.test_ocr_fix_S_and_B)
Sửa lỗi chữ S thành 5 và B thành 8. ... ok
test_separators_hyphens_dots (tests.test_cccd_extractor.TestExtractIdNumber.test_separators_hyphens_dots)
Chuỗi có gạch ngang hoặc dấu chấm. ... ok
test_separators_spaces (tests.test_cccd_extractor.TestExtractIdNumber.test_separators_spaces)
Chuỗi có khoảng trắng giữa các cụm số. ... ok
test_ten_digits_returns_none (tests.test_cccd_extractor.TestExtractIdNumber.test_ten_digits_returns_none)
10 chữ số -> None. ... ok
test_text_with_names_not_corrupted (tests.test_cccd_extractor.TestExtractIdNumber.test_text_with_names_not_corrupted)
Văn bản họ tên hoặc từ ngữ không bị sửa thành số linh tinh. ... ok
test_thirteen_digits_returns_none (tests.test_cccd_extractor.TestExtractIdNumber.test_thirteen_digits_returns_none)
Thừa 1 chữ số (13 số) -> Bắt buộc trả None, KHÔNG cắt xén 12 số đầu. ... ok
test_valid_12_digits (tests.test_cccd_extractor.TestExtractIdNumber.test_valid_12_digits)
Số CCCD chuẩn 12 chữ số. ... ok

----------------------------------------------------------------------
Ran 26 tests in 0.254s

OK
```

---

## 4. Tương thích Hệ thống

- **API Upload cũ (`main.py`)**: Giữ nguyên route `POST /api/v1/upload`, trả về cấu trúc JSON tương thích hoàn toàn.
- **Frontend Flet (`cccd-reader-fe-main/main.py`)**: Đã ánh xạ `FIELD_LABELS["cccd"]` với 2 trường mới:
  - `"id_number"`: `"Số CCCD"`
  - `"expiry_date"`: `"Có giá trị đến"`
- **GPLX Extractor**: Không bị ảnh hưởng (regression tests đều đạt).

---

## 5. Những hạn chế chưa khắc phục & Cần kiểm tra trên ảnh thực tế

| Nội dung | Trạng thái | Ghi chú & Kế hoạch xử lý |
|---|---|---|
| Kiểm tra độ chính xác OCR Số CCCD trên ảnh thật | **NOT_TESTED** | Chưa có tập ảnh test thực tế trong workspace; cần chạy `debug_roi.py` với ảnh mẫu để kiểm tra độ khít của ROI |
| Kiểm tra độ chính xác OCR Ngày hết hạn trên ảnh thật | **NOT_TESTED** | Cần ảnh mẫu mặt trước CCCD chip 2021 thật để tinh chỉnh tọa độ `BOX_EXPIRY_DATE` |
| Hỗ trợ mẫu Thẻ Căn cước (từ 01/07/2024) | **CHƯA HỖ TRỢ** | Hiện extractor chỉ chuyên cho CCCD chip 2021 (Thông tư 06/2021). Cần thêm bộ phân loại mẫu thẻ trước khi mở rộng |
| Xử lý ảnh bị chụp nghiêng/xoay | **CHỜ GIAI ĐOẠN 3** | Hiện tại cắt box theo tọa độ tương đối cố định; sẽ được giải quyết triệt để khi hoàn thành module `document_rectifier` (Giai đoạn 3) |

---
