# AGENTS.md — XLA | Autonomous, Efficient & Safe Multi-Agent Rules

> **Phiên bản:** 2.0 — 2026-11-18  
> **Đặt tại:** `C:\Users\bui nhu duc\XLA\AGENTS.md` (repository root)  
> **Phạm vi:** Mọi agent, worker, verifier và phiên Antigravity làm việc trong repository XLA.  
> **Nguyên tắc:** *Tự thực thi khi được giao; chỉ song song khi có lợi; chia sẻ thay đổi kịp thời; bảo vệ dữ liệu và mã nguồn.*

## 1. Thứ tự ưu tiên và quy trình bắt buộc

1. Tuân thủ yêu cầu cụ thể mới nhất của người dùng, chính sách/quyền công cụ, sau đó là các quy tắc an toàn trong tài liệu này. Không dùng tài liệu này để vượt qua cơ chế xin quyền của môi trường.
2. **Đầu mỗi phiên / task:** đọc `AGENTS.md`; đọc `XLA_COORDINATION.md` nếu tồn tại; kiểm tra `git status --short`, `git branch --show-current`, file đang thay đổi và nhiệm vụ đang mở.
3. Khi nhận yêu cầu **sửa / fix / thêm / refactor / triển khai**, chủ động thực hiện trong phạm vi đã giao, không dừng ở phân tích hoặc xin phép từng chỉnh sửa thông thường.
4. Chu trình chuẩn: **READ → SCOPE → ASSIGN/CLAIM → EDIT → SHARE → TEST → VERIFY → HANDOFF**.
5. `AGENTS.md` là tài liệu **quy tắc cố định**; **KHÔNG** dùng làm nhật ký. Cập nhật tiến độ trong tài liệu điều phối và kênh trao đổi phù hợp.
6. Không báo `PASS`, `DONE`, "đã sửa" hoặc "đã build" nếu không có bằng chứng tương ứng trong filesystem/lệnh thực thi.

## 2. Lựa chọn chế độ làm việc để TỐI ƯU THỜI GIAN

### 2.1 Quyết định theo công việc, không theo số agent tối đa

| Loại nhiệm vụ | Cách triển khai ưu tiên |
|---|---|
| Một lỗi nhỏ, 1–2 file, thay đổi khó chia độc lập | **1 agent**, sửa → test → báo cáo |
| 2 hoặc 3 module/file-set độc lập, kiểm thử riêng được | **`/swarm` với 2–3 implementation workers** và verifier khi phù hợp |
| Tác vụ lớn hơn nhưng ít phụ thuộc, sở hữu file tách được | Chỉ tăng worker nếu lợi ích lớn hơn chi phí điều phối/quota |
| Các thay đổi chồng vào cùng file, schema hay config chung | Chia giai đoạn hoặc **tuần tự**, không ép chạy song song |
| Train/inference/build nặng, tranh chấp CPU/RAM/đường dẫn output | **Tuần tự hóa tác vụ nặng**; cho audit/test nhẹ chạy song song nếu an toàn |

**Mặc định** cho nhiệm vụ lớn trong XLA: **2–3 worker chỉnh sửa + 1 verifier**. Đây là mức khởi đầu để tối ưu tốc độ, **không phải yêu cầu phải tạo đủ số worker**. Nếu nhiệm vụ nhỏ, sử dụng một agent. Nếu nhiều worker gây chậm, quota cao hoặc xung đột, tự giảm số lượng và ghi lý do.

### 2.2 Quy tắc tăng tốc

- Chia theo **output độc lập, package, file ownership và dependency**, không chia máy móc theo số bước.
- Ưu tiên triển khai phần độc lập trước, đồng thời chuẩn bị giao diện/hợp đồng dùng chung sớm.
- Không bắt cả nhóm chờ một worker khi còn việc độc lập có thể tiếp tục.
- Dùng **event-driven updates**, không yêu cầu ghi trạng thái sau từng dòng code hoặc mỗi lệnh terminal.
- Không tạo các vòng hỏi đáp, báo cáo lặp lại, hoặc kiểm thử E2E nặng không cần thiết cho mỗi chỉnh sửa nhỏ.
- Không nhân bản tải YOLO/OCR, train GPU/CPU, hoặc build Flet/PyInstaller đồng thời trên cùng output directory.
- Không tự chạy liên tục/hậu trường khi người dùng không giao nhiệm vụ và không có công cụ hỗ trợ.

### 2.3 Lệnh và giới hạn nền tảng

- Dùng `/swarm <nhiệm vụ>` **khi lệnh có sẵn trong Antigravity**; công cụ sẽ có coordinator nội bộ, worker và thường có verifier. Người dùng **không cần quản lý Main Agent thủ công**.
- Trong Antigravity App, mở giao diện Swarm từ phản hồi coordinator; trong CLI có thể dùng `/swarm board` hoặc `Alt+S` để xem board.
- Worker liên lạc bằng Shared Message Board và `@mentions` nếu nền tảng hỗ trợ. **Không tự tạo /swarm giả hoặc giả vờ các worker đang chạy.**
- Nếu `/swarm` không khả dụng (phiên bản/gói/quota/cấu hình), thông báo đúng giới hạn và triển khai bằng agent/subagent thực sự có sẵn; không khẳng định có đồng bộ realtime nếu môi trường không hỗ trợ.

## 3. XLA: bối cảnh kỹ thuật và dữ liệu PHẢI bảo vệ

- Root dự kiến: `C:\Users\bui nhu duc\XLA`; **xác minh** chứ không tự giả định.
- Backend `cccd-reader-api-main/` (FastAPI, OCR CCCD/GPLX); frontend `cccd-reader-fe-main/` (Flet Windows).
- API v2 thường dùng `POST /api/v2/upload` với chế độ Fast/Accurate và lựa chọn loại tài liệu; trước khi sửa phải kiểm tra contract tại source hiện hành.
- Hướng chuyển sang YOLOv8-Pose cần **xác minh code và checkpoint thật**; không kết luận hệ thống đã hoàn toàn bỏ YOLO Detection chỉ dựa trên tài liệu hoặc tên thư mục.
- Các thí nghiệm VietOCR/CTPN phải cách ly khỏi đường production cho đến khi được yêu cầu tích hợp và có benchmark đủ tin cậy.

**Mặc định KEEP, không xóa hoặc ghi đè:**

- Mọi ảnh ROI; OCR crops; ảnh rectified/warped; OCR debug; ROI calibration; ảnh và log phục vụ benchmark/ground truth.
- `calibrated_rois.py`, `document_rectifier.py`, `document_orientation.py`, các module OCR/Pose còn sử dụng.
- Dataset CCCD/GPLX gốc; dataset Pose; keypoint labels/annotations; ground truth; checkpoint đang dùng hoặc có giá trị phục hồi.
- `models/`, `datasets/`, `debug_output/`, thư mục chứa `roi`, `ocr`, `rectified`, `ground_truth`, `checkpoints` khi chưa audit rõ nội dung.
- User changes chưa commit, secrets, cấu hình triển khai, dữ liệu cá nhân và artifact chưa rõ nguồn gốc.

**Ngoại lệ:** Chỉ xóa file có **phạm vi người dùng đã duyệt rõ ràng**, tốt nhất theo manifest exact paths. Phê duyệt trước đó cho 57 legacy Detection outputs không tự động cho phép xóa những file khác hoặc file mới xuất hiện sau audit. Trước xóa cần kiểm tra đường dẫn thực tế, dependency và bảo toàn hash dữ liệu được bảo vệ. Không được xóa cả thư mục bằng đệ quy chỉ vì tên `runs`, `outputs`, `debug`, `datasets`.

## 4. Parallel file ownership và phòng xung đột

1. Mỗi worker nhận **Task ID**, mục tiêu, `owned_paths`, dependencies, acceptance criteria và risk level.
2. **Một file chỉ có một writer tại một thời điểm**; các agent khác chỉ đọc. Các thư mục có file sinh tự động dùng chung cũng phải có owner.
3. Trước khi sửa, worker đọc trạng thái Git và source mới nhất; không ghi đè changes của người dùng hoặc worker khác.
4. Muốn chỉnh file do agent khác sở hữu: thông báo trên board, yêu cầu chuyển ownership; chỉ sửa sau khi đã được chuyển quyền.
5. File dùng chung như `main.py`, `schemas.py`, `requirements*.txt`, `AGENTS.md`, `XLA_COORDINATION.md` phải được điều phối tuần tự khi nhiều agent có nhu cầu.
6. Nếu các worker có working copy/worktree tách biệt, **phải tích hợp thay đổi về đúng target tree** và xác minh code tồn tại tại đó trước khi test chung. Không tự checkout/switch/rebase/merge trong repo của người dùng nếu không có quyền.
7. Không chạy nhiều task cùng ghi một `dist/`, `build/`, checkpoint, log hoặc cache chung.
8. Markdown là dữ liệu trạng thái **không phải atomic lock**: ownership phải được điều phối bằng công cụ/board/nhà điều phối sẵn có. Nếu không xác minh được quyền ghi độc quyền, tạm dừng sửa file tranh chấp và tiếp tục phần độc lập.

## 5. Đồng bộ LIVE trong lúc sửa — nhanh nhưng đủ thông tin

### 5.1 Ba kênh có mục đích khác nhau

- **Shared Message Board:** trao đổi sự kiện quan trọng trong thời gian chạy, dùng `@mention` đúng người phụ thuộc.
- **`coordination/agents/<worker_id>.md`:** worker có thể ghi nhật ký riêng, chỉ **worker đó** được chỉnh file của mình.
- **`XLA_COORDINATION.md`:** snapshot bền vững gồm Task Board, ownership, contracts, milestones và test evidence; **một writer/scribe** được coordinator chỉ định ghi tuần tự. Không cần cậu chỉ định Main Agent thủ công.

### 5.2 Khi nào PHẢI cập nhật ngay

Chỉ cập nhật khi có **sự kiện có ý nghĩa**, gồm:

- `CLAIMED`: nhận task và claim đường dẫn.
- `CONTRACT_PROPOSED` hoặc `CONTRACT_CHANGED`: thay đổi API/schema/signature/format. Thông báo **trước khi consumer tích hợp**.
- `PARTIAL_READY`: một phần chức năng đã có thực và có thể dùng; chỉ rõ điều gì đã test/chưa test.
- `BLOCKED`: thiếu dependency, lỗi, tranh chấp file hoặc test quan trọng thất bại.
- `TEST_PASS` / `TEST_FAIL`: có kết quả lệnh thực thi thực tế.
- `HANDOFF` / `DONE`: bàn giao và giải phóng ownership.

**Không cần gửi heartbeat định kỳ hoặc cập nhật từng edit.** Giai đoạn kéo dài không có sự kiện mới vẫn tiếp tục thực hiện công việc; worker được nhắc bằng `@mention` khi cần phối hợp.

### 5.3 Mẫu thông báo

```text
[SWARM UPDATE]
Time: <ISO-8601>
Worker/Task: <worker_id> / <task_id>
State: CLAIMED | CONTRACT_CHANGED | PARTIAL_READY | BLOCKED | TEST_PASS | TEST_FAIL | HANDOFF | DONE
Owned files: <paths>
Delivered: <verified change or milestone>
Contract: <change or NONE>
Tests: <exact command + result, or NOT RUN>
Needs from others: <@worker + action, or NONE>
Next: <next action>
```

### 5.4 Worker khác đọc và phản ứng

- Đọc cập nhật khi được `@mention`, khi dependency thay đổi, trước chỉnh sửa interface phụ thuộc, trước integration tests và trước handoff.
- Kiểm tra **code mới thực sự có tại target working tree** và contract đúng trước khi sử dụng.
- Nếu một phần `PARTIAL_READY` chưa được test, có thể tích hợp thận trọng nhưng không tuyên bố đã ổn định.
- Nếu xảy ra xung đột file: báo `BLOCKED` cho phần bị ảnh hưởng, chuyển sang phần độc lập cho tới khi ownership rõ ràng.

## 6. Quy trình cập nhật file điều phối

`AGENTS.md` **không** được dùng làm status log.

`XLA_COORDINATION.md` lưu: Project State, Task Board, File Ownership, Interface/Dependency Contracts, Milestones, Test Evidence, Blockers, Decisions, Handoffs.

- Một writer đã được chỉ định: **đọc bản mới nhất → hợp nhất thay đổi đã kiểm chứng → ghi đúng mục → xác minh không mất update**.
- Chỉ cập nhật snapshot khi **bắt đầu nhiệm vụ, thay đổi ownership/contract, milestone, blocker, kết quả test hoặc bàn giao**; không rewrite file sau từng thông báo nhỏ.
- Không xóa dữ liệu lịch sử đã xác minh hoặc trạng thái đang dùng của phiên khác.
- Người tiếp nhận nhiệm vụ phải đọc lại `XLA_COORDINATION.md` và file trạng thái worker có liên quan, sau đó kiểm tra source code thực tế.
- Nếu file chưa tồn tại: tạo **mẫu khởi tạo**, không điền số liệu PASS hoặc trạng thái giả định.

## 7. Tự sửa và kiểm thử

- Trong phạm vi được giao: read → reproduce (khi cần) → edit → run relevant tests → fix failures → verify → handoff; không dừng ở đề xuất.
- Test nhỏ thuộc worker sở hữu module; verifier chạy diff review và các integration tests cần thiết sau khi các phần liên quan đã hợp nhất.
- Tránh nhân bản cùng bộ E2E OCR thực trên nhiều worker, vì có thể làm nghẽn RAM/CPU và không tăng tốc độ hoàn thành.
- Frontend Flet: unit UI không thay thế kiểm thử cửa sổ chạy từ source; nếu có build `.exe`, kiểm tra bản đóng gói trên Windows.
- FastAPI: kiểm tra import, `/health`, endpoint upload liên quan khi được sửa.
- YOLO/Pose/OCR: đối chiếu output, benchmark, ground truth, timing/RAM khi thay đổi thuật toán; không tự bật model thử nghiệm vào production.
- Cleanup: so sánh manifest, count/hash protected ROI/datasets/checkpoints, Git status và runtime dependency trước/sau.
- Báo cáo `NOT RUN` nếu không có điều kiện kiểm thử, `FAILED` nếu test lỗi, `PARTIAL` nếu chưa đủ nghiệm thu. Không sửa test để che lỗi.

## 8. Giới hạn an toàn và bảo mật

- Không tự checkout/switch branch, reset hard, `git clean -fdx`, force push, rebase/merge, rewrite Git history, commit hoặc push khi chưa được người dùng yêu cầu.
- Không xóa dữ liệu nhạy cảm hoặc file ngoài manifest được duyệt; không dọn `ROI/OCR/Pose/datasets/ground_truth/checkpoints` theo suy đoán.
- Không đưa ảnh CCCD/GPLX, thông tin định danh, logs chứa PII, secrets hoặc checkpoint chưa được phép lên dịch vụ ngoài / Shared Board / Markdown.
- Không tự cài/chuyển dependency phá môi trường, thay API production, mở public network endpoint, thay thiết lập bảo mật hoặc chạy migration không thể đảo ngược ngoài phạm vi đã cho.
- Các thao tác có hậu quả phá hủy, vượt scope, chi phí đáng kể hoặc cần cấp quyền công cụ riêng phải được phê duyệt theo quy trình phù hợp.
- Nếu có xung đột yêu cầu: dừng **phần nguy hiểm**, tiếp tục các phần độc lập an toàn và nêu rõ giới hạn.

## 9. Điều kiện kết thúc và mẫu báo cáo

Một nhiệm vụ chỉ `DONE` khi bản vá ở **đúng target tree**, không còn xung đột ownership, các bài test bắt buộc đã chạy và trạng thái điều phối được cập nhật.

Mỗi worker cung cấp: `Task ID`, `Files changed`, `Summary`, `Contracts`, `Tests (commands + results)`, `Blockers`, `Handoff`.

Verifier kiểm chứng sau tích hợp. Báo cáo cho người dùng bao gồm: các mục `DONE/PARTIAL/FAILED`, ai thực hiện, file đã chỉnh, test evidence, vấn đề còn lại, những thao tác cần quyền người dùng. Không mô tả tính năng chưa thực thi là đã hoàn thành.

## 10. Phạm vi hiệu lực

- Những quy tắc này áp dụng cho các nhiệm vụ XLA tiếp theo **khi agent thực sự đọc được file** từ workspace.
- Một file Markdown không tự tạo agent, không tự gửi thông báo hoặc khóa file. Điều phối/sync realtime chỉ thực sự xảy ra khi tính năng Antigravity và worker đang chạy hỗ trợ.
- Với task mới: **ưu tiên kết quả đúng và tổng thời gian hoàn thành ngắn hơn số lượng worker lớn**.
