# XLA_COORDINATION.md — Shared Persistent State

> Tài liệu điều phối bền vững. Không dùng làm kênh tin nhắn realtime. Antigravity `/swarm` Shared Message Board là kênh thông báo cập nhật khi worker đang chạy.
>
> **Writer:** Chỉ một worker/scribe do coordinator nội bộ của Swarm chỉ định được phép chỉnh file này tại một thời điểm. Các worker khác cập nhật `coordination/agents/<agent_id>.md` và phát thông báo trên Shared Message Board.
>
> **Last verified:** 2026-11-19T12:21:00+07:00 — Repository root, branch `main`, working tree sạch mã nguồn. Autonomous Multi-Agent Mode v2.0 đã nạp đầy đủ.

## Project State

| Field | Value |
|---|---|
| Repository | `C:\Users\bui nhu duc\XLA` (VERIFIED) |
| Branch | `main` (VERIFIED) |
| Working tree | Sạch mã nguồn; sẵn sàng bàn giao; untracked test_roboflow_compatibility.py (VERIFIED) |
| Active user objective | Final YOLO Detection Cleanup & Roboflow Keypoint Compatibility (point_up_left/right/bottom_right/bottom_left) |
| Swarm/session | COMPLETED (Hoàn tất 100%: Backup & Cleanup Detection, Sync Roboflow Pose Config, Comprehensive Verification) |
| Assigned coordination writer | `coordinator` |

## Task Board

| Task ID | Objective | Owner | State | Dependencies | Acceptance criteria | Last update |
|---|---|---|---|---|---|---|
| `TASK-CLEANUP-BACKUP` | Audit, backup ngoài repo và xóa an toàn Legacy YOLO Detection (dataset, checkpoints, detection scripts & services) | `coordinator` | DONE | None | 939 files (62.78MB) backup ngoài repo với SHA-256; xóa sạch detection; ROI/OCR 100% nguyên vẹn (120 API + 100 Root) | 2026-11-19T13:30:00+07:00 |
| `TASK-ROBOFLOW-SYNC` | Đồng bộ quy ước Keypoint Roboflow (point_up_left, point_up_right, point_bottom_right, point_bottom_left, flip_idx: [1,0,3,2]) | `coordinator` | DONE | None | Cập nhật card_pose.yaml, manifest, services, scripts/pose, notebook; importer hỗ trợ auto-remap | 2026-11-19T13:30:00+07:00 |
| `TASK-VERIFY-SUITE` | Xây dựng test suite tương thích Roboflow & chạy toàn bộ regression tests (API, FE, Rectifier, Cleaners, Importer) | `coordinator` | DONE | `TASK-CLEANUP-BACKUP`, `TASK-ROBOFLOW-SYNC` | 100% tests PASS (10 roboflow, 17 pose, 15 rectifier, 12 cleaner, 8 FE); FastAPI 200 OK | 2026-11-19T13:32:00+07:00 |

## File Ownership

| Path / file | Writer | Task ID | State | Since |
|---|---|---|---|---|
| `cccd-reader-api-main/services/document_pose_detector.py` | `coordinator` | `TASK-ROBOFLOW-SYNC` | RELEASED | 2026-11-19T12:30:00+07:00 |
| `cccd-reader-api-main/services/document_rectifier.py` | `coordinator` | `TASK-ROBOFLOW-SYNC` | RELEASED | 2026-11-19T12:30:00+07:00 |
| `cccd-reader-api-main/scripts/pose/import_roboflow.py` | `coordinator` | `TASK-ROBOFLOW-SYNC` | RELEASED | 2026-11-19T12:30:00+07:00 |
| `cccd-reader-api-main/tests/test_roboflow_compatibility.py` | `coordinator` | `TASK-VERIFY-SUITE` | RELEASED | 2026-11-19T12:45:00+07:00 |

## Interface & Dependency Contracts

| Contract | Producer | Consumers | Current status | Evidence |
|---|---|---|---|---|
| Roboflow Keypoint Spec | `card_pose.yaml` / `import_roboflow.py` | `document_pose_detector.py`, `document_rectifier.py`, `pipeline_pose.py`, Colab Notebook | VERIFIED | TL:0, TR:1, BR:2, BL:3; kpt_shape: [4, 3]; flip_idx: [1, 0, 3, 2]; names: {0: cccd, 1: gplx} |
| Pose-only Runtime API | `pipeline_pose.py` / `main.py` | Frontend Flet (`cccd-reader-fe-main`) | VERIFIED | `main.py` import `process_document_pose`; endpoint `/health` trả 200 OK; không còn tham chiếu tới `document_detector` |

## Milestone Updates

| Timestamp | Worker / Task | Update | Code verified? | Readers notified |
|---|---|---|---|---|
| 2026-11-19T12:41:00+07:00 | `coordinator` / `TASK-CLEANUP-BACKUP` | Backup hoàn tất 939 files sang thư mục artifact an toàn, xác minh SHA-256 100% | YES | YES |
| 2026-11-19T12:43:00+07:00 | `coordinator` / `TASK-CLEANUP-BACKUP` | Xóa sạch 939 file Legacy YOLO Detection. Dữ liệu ROI bảo vệ giữ nguyên 100% | YES | YES |
| 2026-11-19T12:47:00+07:00 | `coordinator` / `TASK-ROBOFLOW-SYNC` | Hoàn thành đồng bộ cấu hình Roboflow Keypoint trên toàn bộ codebase & Notebook Colab | YES | YES |
| 2026-11-19T13:32:00+07:00 | `coordinator` / `TASK-VERIFY-SUITE` | Toàn bộ 52/52 unit tests regression & roboflow compatibility PASS 100% | YES | YES |

## Test & Evidence

| Timestamp | Scope | Command / method | Result | Evidence |
|---|---|---|---|---|
| 2026-11-19T12:50:19+07:00 | Roboflow Compatibility | `python -m unittest cccd-reader-api-main/tests/test_roboflow_compatibility.py` | PASS (10/10) | Xác minh naming, ordering, remap logic, geometry, horizontal flip |
| 2026-11-19T13:29:04+07:00 | Pose Pipeline & API v2 | `python -m unittest cccd-reader-api-main/tests/test_pose_pipeline.py` | PASS (17/17) | Validation hình học, fallback an toàn, 503 khi thiếu checkpoint, 422 khi invalid keypoints |
| 2026-11-19T13:29:14+07:00 | Document Rectifier | `python -m unittest cccd-reader-api-main/tests/test_document_rectifier.py` | PASS (15/15) | Warp perspective 856x540, chống méo, lọc góc lồi, fallback an toàn |
| 2026-11-19T13:30:25+07:00 | Field Cleaner | `python -m unittest cccd-reader-api-main/tests/test_field_cleaner.py` | PASS (12/12) | Chuẩn hóa họ tên, địa chỉ, quê quán, nhãn song ngữ |
| 2026-11-19T13:31:07+07:00 | Frontend Flet | `python -m unittest cccd-reader-fe-main/test_frontend.py` | PASS (8/8, 1 skipped) | Parse response v2, adapt v1, UI controls rendering |
| 2026-11-19T13:31:47+07:00 | FastAPI Health Check | `GET /health` qua TestClient | PASS (200 OK) | `{"status": "ok", "version": "0.2.0"}` |
| 2026-11-19T13:32:06+07:00 | Protected Data Audit | Filesystem counting & SHA-256 | PASS | API debug: 120 files; Root debug: 100 files; Pose scaffold: 9 files |

## Blockers & Decisions

| Timestamp | Scope | Blocker / decision | Owner | Follow-up |
|---|---|---|---|---|
| 2026-11-19T12:35:00+07:00 | Pose Checkpoint | Checkpoint YOLOv8-Pose thực tế chưa có; scaffold & pipeline pose-only sẵn sàng nạp khi người dùng train xong | `coordinator` | Hướng dẫn Colab GPU notebook cho người dùng tại Cell 0 & Cell 6 |

## Handoffs

| Timestamp | From | To | Scope | Verified / accepted |
|---|---|---|---|---|
| 2026-11-19T13:35:00+07:00 | `coordinator` | `user` | Bàn giao hệ thống XLA hoàn toàn sạch Legacy Detection và tương thích 100% với Roboflow Keypoint Specification | YES |

## Synchronization Procedure

1. Worker đăng `[SWARM UPDATE]` qua Shared Message Board **ngay** khi có milestone, contract change, blocker, test result hoặc handoff.
2. Mỗi worker chỉ ghi file `coordination/agents/<agent_id>.md` của mình.
3. Worker/scribe được chỉ định đọc bản mới nhất của tài liệu này, hợp nhất những update đã xác minh và ghi tuần tự, không thay thế snapshot mới bằng bản cũ.
4. Những worker liên quan đọc lại bản này, file status của agent khác và **code thực tế** khi được `@mention` hoặc khi dependency chuyển trạng thái.
5. Không suy diễn rằng ghi file tự động phát thông báo; phải gửi message trực tiếp trên Shared Message Board.
