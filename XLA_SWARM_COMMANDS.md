# XLA_SWARM_COMMANDS.md — Lệnh sử dụng Antigravity

> **Dùng cùng** `AGENTS.md` tại root XLA. Đây là bộ prompt mẫu để **copy vào cửa sổ chat Antigravity**, **không phải lệnh PowerShell**.
>
> `/swarm` khả dụng tùy phiên bản và gói tài khoản Antigravity. Khi nhập, chọn slash command từ menu và nhấn `Tab`/`Enter`, rồi gửi mô tả nhiệm vụ.

## A. Một lần sau khi thay AGENTS.md — đồng bộ quy tắc

```text
Đọc toàn bộ AGENTS.md tại root repository XLA và XLA_COORDINATION.md (nếu có). Áp dụng quy tắc v2.0 cho các tác vụ tiếp theo. Kiểm tra branch và working tree, không chỉnh sửa source code nếu chưa có nhiệm vụ cụ thể. Không ghi đè tiến độ cũ. Xác nhận ngắn gọn cách chọn 1 agent vs /swarm và cơ chế đồng bộ LIVE.
```

## B. Công việc nhỏ: 1 agent (NHANH hơn khởi tạo swarm)

```text
XLA TASK: [MÔ TẢ LỖI / THAY ĐỔI]
Đọc AGENTS.md và trạng thái repo. Chủ động sửa code đúng phạm vi, chạy test liên quan, sửa tiếp khi test fail, cập nhật XLA_COORDINATION.md ở các milestone quan trọng. Không tạo swarm khi nhiệm vụ chỉ gồm một thay đổi nhỏ hoặc phụ thuộc vào cùng một file. Không commit/push.
```

## C. Công việc lớn: 2–3 worker song song + verifier

```text
/swarm XLA TASK: [MÔ TẢ TÍNH NĂNG / LỖI / NHÓM CÔNG VIỆC]

Workspace: C:\Users\bui nhu duc\XLA

Đọc AGENTS.md và XLA_COORDINATION.md trước khi nhận việc. Kiểm tra branch và working tree.

Thực thi thực tế, không chỉ tư vấn. Tách nhiệm vụ theo file-set không chồng lấn. Ưu tiên 2–3 implementation workers đồng thời và một verifier khi phù hợp; nếu việc nhỏ, dùng ít worker hơn. Tự giảm worker khi quota, file conflict hoặc tài nguyên máy làm chi phí cao hơn lợi ích.

Mỗi worker tự sửa module được giao, kiểm thử, sửa lỗi; chỉ có một writer trên mỗi file. Không ghi trùng main.py, schemas.py, requirements hoặc output directory. Các contract thay đổi phải thông báo sớm.

Dùng Shared Message Board + @mention để thông báo tại các mốc CLAIMED, CONTRACT_CHANGED, PARTIAL_READY, BLOCKED, TEST_PASS/FAIL, HANDOFF. Các worker phụ thuộc phải đọc cập nhật và đối chiếu source thực tế trước khi tích hợp. Không spam update sau mỗi dòng code.

Mỗi worker chỉ ghi file trạng thái riêng coordination/agents/<worker_id>.md. Một writer được coordinator nội bộ phân công cập nhật XLA_COORDINATION.md theo milestone, không ghi đè dữ liệu cũ. Không dùng AGENTS.md làm nhật ký.

Verifier phải kiểm tra diff, test tích hợp trên code đã hợp nhất tại target working tree. Giữ nguyên ROI/OCR/Pose, dataset, ground truth, checkpoint; không tự đổi branch, reset, commit hoặc push. Chỉ báo cáo DONE có bằng chứng test.

BẮT ĐẦU TRIỂN KHAI.
```

## D. Sửa riêng lỗi frontend Flet / cửa sổ trắng

```text
/swarm XLA – FIX FLET WINDOWS BLANK WINDOW

Đọc AGENTS.md. Sửa lỗi frontend Flet desktop mở cửa sổ trắng. Phân chia tối đa 3 mảng độc lập: (1) lỗi khởi tạo/render source, (2) build/packaging và assets, (3) test/regression. Không để nhiều worker sửa cùng main.py hay build chung dist/. Xác định root cause bằng traceback/log, sửa mã nguồn thực tế, chạy source và bản .exe (nếu có Windows runtime), dùng verifier kiểm tra end-to-end. Cập nhật Shared Board theo sự kiện và XLA_COORDINATION.md theo milestone; không đổi backend OCR nếu không liên quan. Không commit/push.
```

## E. Nghiên cứu thay thế EasyOCR bằng VietOCR/CTPN mà không ảnh hưởng production

```text
/swarm XLA – VIETOCR/CTPN ISOLATED BENCHMARK

Đọc AGENTS.md. Đánh giá khả năng cải thiện OCR bằng VietOCR và CTPN nhưng KHÔNG thay đổi production pipeline. Chia việc độc lập: (1) kiểm toán dataset và ground truth, (2) VietOCR benchmark, (3) CTPN/text-detection và field extraction review; verifier kiểm tra tính nhất quán metric. Không thay đổi checkpoint YOLOv8-Pose hoặc FastAPI contract. Đo CER/WER/field exact match, latency/RAM nếu có ground truth; thiếu ground truth phải báo chưa thể kết luận. Đồng bộ qua Shared Message Board và tài liệu điều phối. Không tự push/commit.
```

## F. Theo dõi trong lúc Swarm đang chạy

**Antigravity App:** mở `Open the Swarm UI` từ phản hồi coordinator.

**Antigravity CLI:** nhập `/swarm board` hoặc nhấn `Alt+S`.

Có thể nhắn trong conversation đang làm coordinator:

```text
Cập nhật trạng thái hiện tại của các worker: task, file ownership, phần đã tích hợp, test result và blocker. Tiếp tục cho các worker tự sửa phần độc lập, chỉ điều phối xung đột và dependency. Không dừng toàn bộ Swarm chỉ để báo cáo.
```

## G. Nhắc Swarm khi có lỗi hoặc thay đổi API giữa chừng

```text
Kiểm tra Shared Message Board. Worker tạo thay đổi API/schema mới phải công bố contract và @mention consumer ngay. Các worker bị ảnh hưởng phải đọc lại code hiện tại, cập nhật phần triển khai phù hợp, test lại. Giải quyết file ownership trước khi sửa file chung. Ghi milestone có xác minh vào XLA_COORDINATION.md, không ghi log trùng lặp.
```

## H. Phạm vi xóa file / cleanup

```text
/swarm XLA – SAFE CLEANUP WITH APPROVED MANIFEST

Đọc AGENTS.md, XLA_COORDINATION.md và cleanup_detection_report.md. Chỉ thực hiện trong đúng phạm vi DELETE manifest đã được người dùng phê duyệt. Có thể kiểm chứng manifest và dependency song song; chỉ một worker được phép xóa file sau bước pre-check. Không xóa ảnh ROI, OCR crop, dataset, checkpoints chính thức hoặc file REVIEW. Sau khi xóa, dùng các worker độc lập đối chiếu hash/count và chạy tests; cập nhật kết quả. Nếu manifest không khớp workspace hiện tại, dừng phần xóa không thể xác minh. Không git clean/reset/commit/push.
```

## I. Lựa chọn nhanh

- **Bug nhỏ / 1 file:** Dùng mục **B**.
- **Tính năng trải qua nhiều module:** Dùng mục **C**.
- **Flet blank window:** Dùng mục **D**.
- **VietOCR/CTPN POC:** Dùng mục **E**.
- **Dọn output Detection đã duyệt:** Dùng mục **H**.
- **Xem worker đang chạy:** Dùng mục **F**.

**Ghi nhớ:** Swarm có chi phí quota lớn hơn một agent; không ép tạo nhiều worker khi không có công việc độc lập. Mẫu lệnh không cấp quyền xóa/sửa vượt quyền công cụ hoặc AGENTS.md.
