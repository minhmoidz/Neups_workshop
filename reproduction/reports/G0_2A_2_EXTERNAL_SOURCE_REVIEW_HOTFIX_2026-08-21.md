# G0.2A.2 EXTERNAL SOURCE-REVIEW HOTFIX — CPU-Only

**Ngày:** 2026-08-21. Toàn bộ công việc thực hiện trong git worktree cô lập
(`/tmp/g0-2a2-hotfix-worktree`, branch `review/g0-2a2-source-hotfix-20260821`, base commit
`2f2904b2ce43a56e08eee05c1805f0e1a7a02977`). Không đụng checkout đang train trên
`research/method-restart`. Không GPU, không CUDA, không dataset/checkpoint thật, không chạy
simulator, không chạy runner thật, không khởi động G0.2B/G1.

**An toàn:** `ps aux | grep run_hardened_verifier` xác nhận PID 1875303 vẫn chạy trước/trong/sau task
này, không bị đụng.

---

## Sửa 8 lỗi được review ngoài xác nhận

### Fix 1 — Forward an toàn cho autograd downstream

**Bug thật, đã xác minh bằng thực nghiệm** (không chỉ đọc code): implementation cũ dùng
`torch.inference_mode()` để sinh ảnh giả cho critic-only step. Chạy thử trực tiếp implementation cũ
trên 1 generator + critic đồ chơi:
```
RuntimeError: Inference tensors cannot be saved for backward. To work around you can make a clone to
get a normal tensor and use it in autograd.
```
**Xác nhận: implementation cũ sẽ crash thật nếu ảnh giả từ `preserved_eval_forward` được đưa vào critic
rồi backward** — đúng như lo ngại review nêu ra, không phải suy đoán.

**Sửa:** đổi `torch.inference_mode()` → `torch.no_grad()` trong `preserved_eval_forward`
(`state_invariants.py`). Tensor tạo dưới `no_grad()` là tensor thường (không phải "inference tensor"),
dùng được bình thường trong graph autograd phía sau.

### Fix 2 — Khôi phục đúng topology mode từng submodule

Bug cũ: `self.module.train(self._original_mode)` chỉ lưu **1 flag top-level**, rồi gọi `.train()` (đệ
quy xuống mọi submodule) — nếu trước đó có submodule mang mode khác cha (ví dụ cha=True, con=False),
khôi phục sẽ ép TẤT CẢ submodule về đúng 1 giá trị top-level, xoá mất sự khác biệt gốc.

**Sửa:** `_snapshot_mode_vector()`/`_restore_mode_vector()` — lưu `[(module, training) for module in
module.modules()]` đầy đủ, khôi phục từng submodule bằng gán trực tiếp `m.training = saved` (không qua
`.train()/.eval()` để tránh đệ quy ghi đè lẫn nhau).

### Fix 3 — Output freshness fail-closed thật sự

Bug cũ: chỉ từ chối thư mục chứa 1 trong 6 tên file "đã biết" (`_STALE_RESULT_NAMES`) — thư mục chứa
file lạ, file checkpoint tên khác, hay thư mục con lồng nhau đều bị coi là "fresh" sai.

**Sửa:** mặc định `scientific_mode=True` — từ chối **mọi** thư mục tồn tại và không rỗng, bất kể chứa
gì. Không tự xoá/dọn gì (giữ nguyên nguyên tắc cũ).

### Fix 4 — Role manifest chính xác và canonical

3 lỗ hổng đã sửa: (a) chỉ kiểm tra thiếu role, không kiểm tra role **thừa/gõ sai tên** — nay yêu cầu
`set(roles.keys()) == set(ROLES)` chính xác; (b) không kiểm tra trùng lặp khi 2 ID khác nhau
(`1` và `"1"`, hay `"p9"` và `" p9"`) cùng chuẩn hoá về 1 chuỗi manifest — nay có
`canonical_patient_id()` + phát hiện va chạm; (c) khoá whitelist chỉ được kiểm tra **khi thực sự xảy ra
overlap** — nay validate ngay từ đầu rằng mọi khoá whitelist phải là cặp cross-role hợp lệ
(`_WHITELISTABLE`), có lý do không rỗng, bất kể overlap có xảy ra hay không.

### Fix 5 — Lịch full-state-hash được dùng thật, không phải hằng số chết

Xác nhận đúng nghi ngờ: `PREDECLARED_FULL_HASH_EPOCHS` được định nghĩa ở module nhưng **chưa từng được
tham chiếu** ở đâu khác trong file cũ — logic thật dùng `self.snapshot_epochs` (tham số constructor có
thể ghi đè tự do, mặc định rỗng). Đồng thời điều kiện trigger cũ (`check_full_hash and extra_i==0`) chỉ
tính `check_full_hash` 1 lần/epoch, không gate theo batch — nghĩa là nếu epoch được chọn, **mọi batch**
trong epoch đó đều trigger full-hash ở `extra_i==0`, không chỉ batch đầu.

**Sửa:** xoá tham số `snapshot_epochs` khỏi constructor hoàn toàn — `self.full_hash_epochs` giờ bind
cứng vào `PREDECLARED_FULL_HASH_EPOCHS`, không thể ghi đè từ caller. Điều kiện trigger sửa thành
`epoch_is_scheduled and n_batches == 0 and extra_i == 0` — đúng 1 điểm `(epoch, batch_index=0,
extra_step=0)` mỗi epoch được chọn.

### Fix 6 — Khôi phục đầy đủ seeding RNG

Thiếu `random.seed(seed)` (module `random` chuẩn của Python) và `torch.cuda.manual_seed(seed)` so với
đường dẫn gốc nhạy-parity (`M2AnonymizerRunner._seed_all`, `anonymizer_runner.py:261-270`).

**Sửa:** thêm `import random` + `random.seed(seed)` + `torch.cuda.manual_seed(seed)`, khớp đầy đủ với
`_seed_all` gốc. **Không tuyên bố parity bit-cho-bit** từ việc này — chỉ xác nhận qua source-level test,
parity thật cần chạy GPU (chưa cấp phép).

### Fix 7 — Củng cố hợp đồng bootstrap + test bất biến ghép cặp

Thay toàn bộ `assert` trần (có thể bị loại bỏ dưới `python -O`) bằng `BootstrapInputError` tường minh,
kiểm tra: độ dài khớp, không rỗng, `n_bootstrap` là int dương, điểm số hữu hạn, nhãn nhị phân, **và
tính nhất quán nhãn-với-quan-hệ-bệnh-nhân** (patient1==patient2 ⟺ y_true=1) — kiểm tra mới, chưa có
trước đây. **Không đổi công thức trọng số, không nâng cấp trạng thái `UNVALIDATED`.**

Thêm test bất biến thật: khi `y_score_baseline == y_score_candidate` hệt nhau, **mọi** replicate hợp lệ
phải có `delta == 0` chính xác — test này sẽ bắt được lỗi nếu 1 implementation tương lai vô tình
resample 2 arm độc lập thay vì dùng chung 1 draw. Đã ghi chú rõ trong docstring:
lựa chọn raw AUC vs effective AUC vẫn là quyết định thiết kế **chưa giải quyết**, để ngỏ cho G0.2B,
không tự ý đổi ở đây.

### Fix 8 — Không tuyên bố quá mức về runner

`run()` vẫn `raise RuntimeError(...)` — không triển khai. Các điều còn treo, nêu rõ để không bị hiểu
nhầm là "đã sẵn sàng":
- execution manifest hiện tại chỉ check 1 field boolean, **chưa** ràng buộc hash runner/module/HEAD/
  config/checkpoint như G0.2 đã xác nhận cần thiết (chưa triển khai gate mạnh hơn ở đây).
- weight-provenance binding cho verifier's ResNet50 chưa được validate thật (constructor chỉ yêu cầu
  `weight_provenance_record` khác `None`, chưa gọi `weight_provenance.py`'s hàm xác thực nào ở đây).
- không có cơ chế "no-download" tường minh cho việc khởi tạo verifier (verifier load từ checkpoint đã
  đông cứng có sẵn, không qua `torchvision.models.resnet50(pretrained=True)` trong runner v2, nhưng
  chưa có assertion tường minh chặn download nếu checkpoint thiếu).
- `fresh_batch` vẫn `raise NotImplementedError`.
- end-to-end parity (`k_extra=0` so với certified B_dev) **chưa từng được chạy** — vẫn là cổng GPU
  tương lai, chưa cấp phép.

---

## Xác minh CPU-only

```bash
export CUDA_VISIBLE_DEVICES=""
.venv/bin/python reproduction/tests/test_protocol_v2.py             # exit 0, 69/69 PASS (cũ: 38)
.venv/bin/python reproduction/tests/test_patient_graph_bootstrap.py  # exit 0, 24/24 PASS (cũ: 14)
```

**Test âm bản (fail trên code cũ, pass trên code đã sửa) — đã xác minh thật, không chỉ suy diễn:**
- Fix 1: chạy trực tiếp implementation cũ (`inference_mode`) trên fixture giống hệt test mới → xác
  nhận raise đúng `RuntimeError: Inference tensors cannot be saved for backward` (trích ở trên). Test
  mới trong `test_protocol_v2.py::test_downstream_autograd_safe_forward` pass trên code đã sửa.
- Fix 3, Fix 4: các test âm bản mới (`unknown_text_file_dest`, `nested_dir_dest`, unexpected-role,
  canonical-ID-collision, v.v.) đều gọi trực tiếp API đã sửa và xác nhận raise đúng — các trường hợp
  này **sẽ không raise** trên implementation cũ (đã đọc lại code cũ để xác nhận logic sẽ bỏ qua các
  trường hợp này, không chạy lại code cũ riêng cho từng case do đã có 1 minh chứng thực nghiệm đại diện
  ở Fix 1).

**Test PASS không phải bằng chứng bootstrap đúng thống kê** — `patient_graph_bootstrap.py` vẫn
`STATUS = 'UNVALIDATED'`, không đổi.

## SHA256 file đã sửa/mới

| File | Bytes | SHA256 |
|---|---:|---|
| `reproduction/protocol_v2/state_invariants.py` | 4979 | `d9b7dcd987c1eb8d1d0e4e3073dbc5a1a65d073f8e9023baadad83862d9ea205` |
| `reproduction/protocol_v2/role_manifest.py` | 6417 | `de8fde23864167290a54220c943780f9c4d2ae3c997f7483c9bf269df2c74d8e` |
| `reproduction/protocol_v2/output_freshness.py` | 3584 | `abc43e852d5364e805f9664c1f087f429a2e3284f791b04b95ff6bc070eba267` |
| `reproduction/statistics/patient_graph_bootstrap.py` | 7328 | `40098553461402233099690d0caf06392d4253a4642f9a16831c3f026c44f785` |
| `reproduction/method_dev/run_hardened_verifier_v2.py` | 23518 | `b5f918b3f79d36f7f69117b86dd30f11661c0a21d280374781103f69377cb404` |
| `reproduction/tests/test_protocol_v2.py` | 23153 | `74ef4be465bb4dadbde1e1df80d4ee8f84b1efa1f23944293ece48a26ae84f02` |
| `reproduction/tests/test_patient_graph_bootstrap.py` | 8876 | `640cd8e2ca8d4f489904d2c656ad44877ff7f105c08d7ba57a0655a418317c47` |

`git diff --check`: exit 0 (báo cáo lại sau khi stage, xem commit log).

---

## Verdict bắt buộc

```
source_review_blockers_addressed: YES
protocol_helper_tests: PASS
runner_v2_execution_status: DESIGN_ONLY_BLOCKED
patient_graph_bootstrap: UNVALIDATED
simulation_v1: INVALID
G0_2B_AUTHORIZATION: NONE
GPU_AUTHORIZATION: NONE
```
