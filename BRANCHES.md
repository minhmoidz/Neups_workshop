# BRANCHES.MD — BẢN ĐỒ NHÁNH (chính sách MỘT NHÀN HOẠT ĐỘNG)

Cập nhật: 2026-08-23. Quyết định hiện hành: **toàn bộ công việc chạy trên MỘT
nhánh duy nhất** cho tới khi có kết quả method ổn định; phân loại nhánh còn lại
chỉ để tra cứu và chống nhầm.

---

## 🟢 NHÁNH LÀM VIỆC DUY NHẤT

`review/p0-runner-attacker-loop-20260823`

Chứa TẤT CẢ: protocol P0_2_3 · attacker_loop · screen/bridge results ·
prereg I_M2 · strategy docs · **code V2** (AgentV2, UNetAtt, configs,
FeatureConsistencyLoss, eval_v2) · BRANCHES.md này.

Mọi thay đổi mới (code V2, phân tích, báo cáo) commit tại đây.
Lưu ý vận hành: trong giai đoạn chuyển tiếp, tiến trình training V2 đang đọc
từ bản sao working-tree của `research/method-restart` local — KHÔNG switch
branch ở đó; các file mới được đồng bộ sang nhánh này bằng copy+commit như đã
làm tại commit tích hợp này.

## 🔵 SNAPSHOT ĐÓNG BĂNG

`research/method-restart` (local) — chụp lại trạng thái trước khi hợp nhất;
sẽ được đồng bộ/xóa quyết định sau khi V2-Uinit hoàn tất. KHÔNG làm việc trên
nhánh này.

## ⚪ BẰNG CHỨNG ĐÓNG BĂNG (read-only)

`review/p0-canonical-protocol-20260821` · `review/p0-2-*` ×4 · `review/g0-2*` ×3

## 🔒 CÁCH LY — TUYỆT ĐỐI KHÔNG MERGE

`research/method-restart-p0p1-review` (`bf43d30`) — chứa nội dung đã bị thu hồi.

## ⚫ CERTIFIED / LEGACY — không đụng

`main` · `archive/legacy-main-20260815` · `promotion/m14c3-clean` ·
`audit/m2-final-certification` · `original-upstream` (`29245d1` code gốc).

---

## QUY TẮC

1. Commit mọi thứ mới vào nhánh làm việc duy nhất ở trên.
2. Cấm merge nhánh cách ly; cấm force-push; cấm commit thẳng `main`.
3. Kết quả khoa học đọc tại: `reproduction/p0_bridge/runs_*/*.json` +
   `reproduction/reports/P0_*`.
4. Worktree `/tmp/*` là vật dụng tạm.
