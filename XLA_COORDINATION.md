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
| Working tree | Sạch mã nguồn; untracked coordination files only (VERIFIED) |
| Active user objective | Kích hoạt Autonomous Multi-Agent Mode v2.0; sẵn sàng nhận nhiệm vụ |
| Swarm/session | READY (Hỗ trợ 1 agent cho task nhỏ, 2–3 worker + verifier cho task lớn) |
| Assigned coordination writer | `coordinator` |

## Task Board

| Task ID | Objective | Owner | State | Dependencies | Acceptance criteria | Last update |
|---|---|---|---|---|---|---|
| — | Chờ giao việc | — | NOT STARTED | — | — | — |

## File Ownership

| Path / file | Writer | Task ID | State | Since |
|---|---|---|---|---|
| — | — | — | UNCLAIMED | — |

## Interface & Dependency Contracts

| Contract | Producer | Consumers | Current status | Evidence |
|---|---|---|---|---|
| — | — | — | UNVERIFIED | — |

## Milestone Updates

| Timestamp | Worker / Task | Update | Code verified? | Readers notified |
|---|---|---|---|---|
| — | — | Chưa có dữ liệu thực thi | NO | — |

## Test & Evidence

| Timestamp | Scope | Command / method | Result | Evidence |
|---|---|---|---|---|
| — | — | — | NOT RUN | — |

## Blockers & Decisions

| Timestamp | Scope | Blocker / decision | Owner | Follow-up |
|---|---|---|---|---|
| — | — | — | — | — |

## Handoffs

| Timestamp | From | To | Scope | Verified / accepted |
|---|---|---|---|---|
| — | — | — | — | — |

## Synchronization Procedure

1. Worker đăng `[SWARM UPDATE]` qua Shared Message Board **ngay** khi có milestone, contract change, blocker, test result hoặc handoff.
2. Mỗi worker chỉ ghi file `coordination/agents/<agent_id>.md` của mình.
3. Worker/scribe được chỉ định đọc bản mới nhất của tài liệu này, hợp nhất những update đã xác minh và ghi tuần tự, không thay thế snapshot mới bằng bản cũ.
4. Những worker liên quan đọc lại bản này, file status của agent khác và **code thực tế** khi được `@mention` hoặc khi dependency chuyển trạng thái.
5. Không suy diễn rằng ghi file tự động phát thông báo; phải gửi message trực tiếp trên Shared Message Board.
