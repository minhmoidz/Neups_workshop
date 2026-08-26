# BRANCHES.MD — BẢN ĐỒ NHÁNH CHÍNH THỨC (single source of truth)

Cập nhật: 2026-08-26. Mọi câu hỏi "đọc/commit/push ở đâu, nhánh gì" tra cứu tại đây.

---

## A. NHÁNH HOẠT ĐỘNG (daily work)

| Nhánh | HEAD | Vai trò | Quy tắc |
|---|---|---|---|
| `research/method-restart` | `a53d352` | **Method development**: code V2 (AgentV2, attention UNet, configs), audit fixes F1/F2, Direction C (U-checkpoint continuation) | ✅ Fork đã được hợp nhất bằng merge commit `e776897` (2026-08-26) — local và remote cùng HEAD, push/pull bình thường |
| `review/p0-runner-attacker-loop-20260823` (remote, @ `2952abd`) | — | **P0 harness chính thức**: protocol P0_2_3, attacker_loop, screen/bridge results, prereg I_M2 | Mọi thay đổi hạ tầng P0 commit tại đây |

## B. BẰNG CHỨNG ĐÓNG BĂNG (read-only, đã external-review)

| Nhánh | Nội dung |
|---|---|
| `review/p0-canonical-protocol-20260821` | Protocol lock ban đầu (local) |
| `review/p0-2-external-source-review-20260822` | P0.2 implementation (protocol `b63f98af…`) |
| `review/p0-2-1-source-closeout-20260822` | P0.2.1 hotfix (protocol `096aeeb7…`) |
| `review/p0-2-2-manifest-artifact-integrity-20260823` | P0.2.2 integrity (protocol `528da8b4…`) |
| `review/g0-2*` ×3 | Audit chuỗi G0.2 lịch sử |

## C. CÁCH LY — TUYỆT ĐỐI KHÔNG MERGE

| Nhánh | Lý do |
|---|---|
| `research/method-restart-p0p1-review` (`bf43d30`) | Chứa báo cáo P0-P1 có phần **đã bị thu hồi** (tham chiếu frozen-eval material). Chỉ tồn tại làm lịch sử audit. Erratum/P0.3 nằm ở nhánh runner-loop |

## D. CERTIFIED / LEGACY — không đụng

`main` · `archive/legacy-main-20260815` · `promotion/m14c3-clean` ·
`audit/m2-final-certification` · `original-upstream` (`29245d1` — code gốc PriCheXy-Net, mốc so sánh)

---

## QUY TẮC TRÁCH NHẶM

1. **Kết quả khoa học P0/paper** → đọc ở `review/p0-runner-attacker-loop-20260823`
   (`runs_screen/`, `runs_im2/`, reports P0_*, prereg).
2. **Code method mới (V2)** → `research/method-restart` local. Khi một run V2 được
   chứng nhận hữu ích: merge `origin/research/method-restart` (strategy docs) vào
   local bằng merge commit ghi chú, RỒI push — biến remote thành fast-forward.
3. **Cấm**: merge `*-p0p1-review`; force-push bất kỳ nhánh nào; commit trực tiếp
   lên `main`.
4. Worktree `/tmp/*` là vật dụng tạm — xóa được khi phase tương ứng đóng.
