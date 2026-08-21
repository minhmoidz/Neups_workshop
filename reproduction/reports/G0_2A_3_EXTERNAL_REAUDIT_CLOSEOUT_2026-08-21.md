# G0.2A.3 EXTERNAL RE-AUDIT CLOSEOUT — Three Remaining Corrections

**Ngày:** 2026-08-21. Worktree cô lập `/tmp/g0-2a3-reaudit-worktree`, branch
`review/g0-2a3-reaudit-closeout-20260821`, base commit `7b00c5c1a2fb1d1c21fa6a3e66c98c222c56c376`.
Không đụng checkout đang train trên `research/method-restart`. CPU-only, synthetic-only, không GPU,
không chạy runner/simulator/thí nghiệm nào.

**An toàn:** `ps aux | grep run_hardened_verifier` xác nhận PID 1875303 vẫn chạy trước/trong/sau task,
không bị đụng.

---

## Correction 1 — Canonical identities phải chi phối overlap liên-role

**Bug xác nhận thật**: overlap check cũ (`role_manifest.py`, dòng `overlap = roles[a] & roles[b]`) so
sánh trên **raw Python set**. `generator_train={1}` và `generator_select={"1"}` **không giao nhau** như
raw set (`1 != "1"` trong Python) dù cả 2 cùng chuẩn hoá về `"1"` — overlap thật bị bỏ sót hoàn toàn.

**Sửa**: tính `canonical_sets[role] = {canonical_patient_id(p) for p in patients}` cho mọi role (tái sử
dụng `_check_canonical_collisions` đã có để vừa validate uniqueness trong-role vừa lấy canonical set),
rồi mọi so sánh overlap liên-role (`_HARD_FORBIDDEN`, `_WHITELISTABLE`) dùng `canonical_sets[a] &
canonical_sets[b]` thay vì raw set. `build_manifest()` giờ serialize đúng `canonical_patient_id(p)` cho
mọi ID, và `patient_count` tính từ kích thước tập canonical đã validate.

5 test âm bản mới, đều PASS: (1) `{1}` vs `{"1"}` bị từ chối (hard-forbidden); (2) `" p9"` vs `"p9"`
qua `locked_confirm` bị từ chối; (3) ID tương đương canonical qua cặp whitelistable đòi hỏi lý do tường
minh, có lý do thì qua; (4) `" p9 "` serialize đúng thành `"p9"` trong manifest,
`patient_count` phản ánh đúng tập đã dedupe; (5) mọi test collision trong-role cũ vẫn PASS nguyên vẹn
(không đổi hành vi).

## Correction 2 — GeneratorStateGuard phải kiểm tra đầy đủ mode topology

**Bug xác nhận thật**: `GeneratorStateGuard` cũ định nghĩa **lặp lại** ngay trong
`run_hardened_verifier_v2.py` (không dùng chung với `state_invariants.py`), chỉ lưu **1 flag
top-level** (`self.generator.training`), và verification được gọi **thủ công sau `with`**
(`guard.verify_unchanged()`) — nghĩa là nếu có exception trong khối `with`, verification **không bao
giờ chạy** (code sau `with` không tới được).

**Sửa**: chuyển `GeneratorStateGuard`/`StateInvariantViolation` thành **implementation thật duy nhất**
trong `state_invariants.py`, dùng `_snapshot_mode_vector()`/so sánh đầy đủ (tái sử dụng hạ tầng từ
Fix 2 trước). Verification chuyển hẳn vào `__exit__`:
```python
def __exit__(self, exc_type, exc_val, exc_tb):
    reason = self._drift_reason()
    if reason is None:
        return False  # không có drift: exception gốc (nếu có) truyền nguyên vẹn
    raise StateInvariantViolation(reason) from exc_val
```
`run_hardened_verifier_v2.py` giờ chỉ **import** guard này, xoá bỏ định nghĩa trùng lặp và lệnh gọi
`guard.verify_unchanged()` thủ công. Đồng thời sửa tài liệu cũ còn ghi "inference_mode" — nay đúng
"no_grad" (Fix 1, G0.2A.2).

5 test mới đều PASS: (1) exit bình thường, không đổi gì → pass; (2) top-level mode không đổi nhưng
**child mode đổi** → raise (chỗ mà check top-level-only sẽ bỏ sót); (3) exception không liên quan,
không có drift → **truyền nguyên vẹn** (không bị thay bằng `StateInvariantViolation`); (4) vừa có drift
buffer vừa có exception không liên quan → raise `StateInvariantViolation` với exception gốc được
**chain** (`__cause__`); (5) source thật của runner được xác nhận: import đúng guard, không định nghĩa
lớp trùng lặp, không còn gọi `verify_unchanged()` thủ công, không còn nhắc "inference_mode".

## Correction 3 — Chứng minh 1 draw duy nhất mỗi bootstrap replicate

Giữ nguyên test zero-delta cũ nhưng **sửa lại mô tả**: chỉ là bất biến hành vi (behavioral invariant),
**không phải bằng chứng độc lập** cho việc dùng chung 1 draw — 1 implementation bệnh lý về lý thuyết
vẫn có thể thoả điều kiện này bằng cách khác.

**Test instrument mới** (`test_exactly_one_draw_per_replicate_instrumented`): monkeypatch trực tiếp
`patient_graph_bootstrap.draw_patient_multiplicities` và `.weighted_auc` **ở cấp module** (vì
`patient_graph_bootstrap_paired` gọi chúng như tên global thuần, resolve tại thời điểm gọi — patch cấp
module ảnh hưởng đúng lời gọi bên trong hàm đó). Xác nhận trực tiếp từ chuỗi lời gọi thật:
- `draw_patient_multiplicities` gọi đúng **1 lần/replicate**.
- `weighted_auc` gọi đúng **2 lần/replicate** (baseline, candidate).
- 2 lời gọi trong cùng 1 replicate nhận **đúng cùng 1 mảng weights** (so sánh `np.array_equal`).
- Monkeypatch được khôi phục trong `finally`, xác nhận lại bằng identity check sau khi chạy.

**Không đổi công thức trọng số, không nâng cấp `UNVALIDATED`, không chạy lại simulator v1 (đã xác nhận
`INVALID` ở G0.2A.1).**

---

## Xác minh CPU-only

```bash
export CUDA_VISIBLE_DEVICES=""
.venv/bin/python reproduction/tests/test_protocol_v2.py             # exit 0, 85/85 PASS
.venv/bin/python reproduction/tests/test_patient_graph_bootstrap.py  # exit 0, 29/29 PASS
python -c "import torch; print(torch.cuda.is_initialized())"        # False
```
`git diff --check`: sạch (xem log commit).

## Phân biệt rõ 4 trạng thái (không gộp làm 1)

- **Source/helper correctness**: 3 correction đều verify bằng test thật (không chỉ đọc code), PASS.
- **CPU test status**: 114/114 (85+29) PASS, CUDA không khởi tạo.
- **Statistical validation status**: `patient_graph_bootstrap` vẫn `UNVALIDATED`; `simulation_v1` vẫn
  `INVALID` (đã xác nhận ở G0.2A.1, không chạy lại).
- **Runner execution readiness**: `run()` vẫn raise stub, execution-manifest gate vẫn chặn — **chưa sẵn
  sàng chạy thật**, chỉ đã sửa đúng các lỗ hổng source được review chỉ ra.

## SHA256 file đã sửa

| File | Bytes | SHA256 |
|---|---:|---|
| `reproduction/protocol_v2/role_manifest.py` | 7545 | `69a4458adabf0aabcd35a7f12ef6a1efe2079bf48c946c6a68bffc2b17d5e99d` |
| `reproduction/protocol_v2/state_invariants.py` | 8411 | `14d2127f543fd80c2c78c1c50dfce69eb6badeecf042d95fd4b7664aba320b5e` |
| `reproduction/method_dev/run_hardened_verifier_v2.py` | 22627 | `2f3136e2d37780a0342eb317d331b31d298c1c9d5589120e0e9e03ca183f078f` |
| `reproduction/tests/test_protocol_v2.py` | 29768 | `0383e0b5ae354b50fac54b52e47242621814e5ef44b4c463a4715144cb3aeca2` |
| `reproduction/tests/test_patient_graph_bootstrap.py` | 12353 | `de124ba58071348b9e7c52bc9cfa1184b921677788bbf404ae537e8ac09313a2` |

---

## Verdict bắt buộc

```
G0_2A_3_verdict: PASS
canonical_cross_role_overlap: VERIFIED
full_mode_topology_guard: VERIFIED
paired_single_draw_contract: VERIFIED
runner_v2_execution_status: DESIGN_ONLY_BLOCKED
patient_graph_bootstrap: UNVALIDATED
simulation_v1: INVALID
external_reaudit_status: AWAITING_EXTERNAL_REVIEW
G0_2B_AUTHORIZATION: NONE
GPU_AUTHORIZATION: NONE
```
