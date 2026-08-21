# G0.2 IMPLEMENTATION GATE — CPU-Only Protocol Scaffolding and Statistical Validation

**Ngày:** 2026-08-21. Toàn bộ công việc: CPU-only, dữ liệu tổng hợp (synthetic) hoặc thao tác thuần
code — không GPU, không CUDA, không dataset/checkpoint thật, không truy cập artifact TEST nào.

---

## 1. File đã tạo (chính xác)

```
reproduction/protocol_v2/__init__.py
reproduction/protocol_v2/state_invariants.py
reproduction/protocol_v2/deterministic_loader.py
reproduction/protocol_v2/role_manifest.py
reproduction/protocol_v2/output_freshness.py
reproduction/protocol_v2/weight_provenance.py
reproduction/statistics/__init__.py
reproduction/statistics/patient_graph_bootstrap.py
reproduction/statistics/simulate_bootstrap_coverage.py
reproduction/method_dev/run_hardened_verifier_v2.py
reproduction/tests/__init__.py
reproduction/tests/test_protocol_v2.py
reproduction/tests/test_patient_graph_bootstrap.py
reproduction/reports/G0_2_IMPLEMENTATION_REPORT_2026-08-21.md   (chính báo cáo này)
```

Không tạo file nào ngoài danh sách trên. Không manifest/split dữ liệu thật nào được tạo.

## 2. Git status trước/sau

**Trước:** `HEAD 34bca1f74662275ced0418ea183a3d9d5ef81f88`, branch `research/method-restart`,
`git status --short` chỉ có 4 file untracked không liên quan (`.npmrc`, `node_modules/`,
`package-lock.json`, `package.json`).

**Sau:** `git status --short` **không đổi** — vẫn chỉ đúng 4 file untracked đó. Mọi file mới đều nằm
trong `reproduction/` (gitignore toàn bộ thư mục này, đã xác nhận từ đầu phiên làm việc) → không có
thay đổi nào git nhìn thấy trên bất kỳ file governed nào.

## 3. Quan sát tiến trình đang chạy (không can thiệp)

```
PID 1875303, run_hardened_verifier.py --arm B_dev --k_extra_verifier_steps 2 --seed 42 --tag k3
Trạng thái: đang chạy trước và sau toàn bộ công việc G0.2, không bị kill/signal/attach.
```

## 4. Ma trận code-to-requirement

| Yêu cầu (§ trong prompt) | File | Đạt |
|---|---|:---:|
| 3.1 Canonical tensor-state hash | `state_invariants.py::canonical_tensor_state_hash` | ✅ |
| 3.2 Buffer-only hash | `state_invariants.py::buffer_only_hash` | ✅ |
| 3.3 Parameter version signature | `state_invariants.py::parameter_version_signature` | ✅ |
| 3.4 Preserved eval-forward context manager | `state_invariants.py::preserved_eval_forward` | ✅ |
| 4 Deterministic loader scaffold | `deterministic_loader.py` | ✅ |
| 5 Role-manifest validator | `role_manifest.py` | ✅ |
| 6 Output freshness + weight provenance | `output_freshness.py`, `weight_provenance.py` | ✅ |
| 7 Direction B v2 (write-only) | `run_hardened_verifier_v2.py` | ✅ (AST-checked, không import/chạy) |
| 8 Candidate patient-graph bootstrap | `patient_graph_bootstrap.py` | ✅ (status UNVALIDATED, giữ nguyên) |
| 9 Synthetic coverage/type-I simulation | `simulate_bootstrap_coverage.py` | ✅ chạy được, ⚠️ kết luận không dùng được — xem §14 |
| 10 Test suite | `test_protocol_v2.py`, `test_patient_graph_bootstrap.py` | ✅ 52/52 PASS |

## 5. Lệnh test đã chạy / kết quả

```bash
export CUDA_VISIBLE_DEVICES=""
.venv/bin/python reproduction/tests/test_protocol_v2.py            # 38/38 PASS
.venv/bin/python reproduction/tests/test_patient_graph_bootstrap.py # 14/14 PASS
git diff --check                                                    # exit 0, sạch
```
Không chạy full test suite của repo (đúng yêu cầu, tránh chạm đường dẫn scientific đã governed).

## 6. Bằng chứng không có CUDA/model/data path nào được khởi tạo

- `CUDA_VISIBLE_DEVICES=""` đặt tường minh trước khi chạy test.
- `test_protocol_v2.py` dòng đầu: `assert not torch.cuda.is_initialized()` — assertion PASS.
- `run_hardened_verifier_v2.py` **chỉ được AST-parse** (`ast.parse`, không `import`), xác nhận qua
  lệnh riêng biệt, không thông qua test suite.
- Không dataset thật (`LazyPairDataset`, `SiameseDataset`), không checkpoint thật, không
  `image_pairs_*.txt` nào được mở trong bất kỳ file mới nào — toàn bộ test dùng `ToyBNNet`,
  `ToySeqDataset`, file tạm (`tempfile.TemporaryDirectory`) chứa byte giả.

## 7. Kết quả test state-invariant (§3)

12/12 PASS, bao gồm: xác nhận lại đúng bug gốc (train()+no_grad() vẫn làm BN buffer đổi — tái hiện
đúng H0.1 trên mạng đồ chơi), xác nhận `preserved_eval_forward` giữ nguyên state hash toàn phần, khôi
phục đúng mode kể cả khi có exception, hash phát hiện đúng thay đổi ở cả parameter và buffer, xử lý
đúng buffer nhiều dtype khác nhau (float32 `running_mean` + int64 `num_batches_tracked`) không lỗi.

## 8. Kết quả test deterministic loader (§4)

7/7 PASS: cùng seed → cùng thứ tự epoch-0 giữa 2 sampler độc lập; seed khác → thứ tự khác; **tiêu thụ
RNG toàn cục tuỳ ý trước khi iterate không ảnh hưởng thứ tự** (đây chính là điểm sửa cho gap đã xác
nhận ở `G0_1_PROTOCOL_REPAIR_SPEC_2026-08-21.md` §2.1); tạo lại loader tái lập đúng chuỗi 3 epoch; hash
thứ tự ngữ nghĩa đổi khi ánh xạ sample-id đổi; validation loader giữ tuần tự.

## 9. Kết quả test role-manifest (§5)

11/11 PASS: case disjoint hợp lệ; cả 6 overlap bị cấm đều bị từ chối đúng; overlap được whitelist kèm
lý do thì qua; overlap được whitelist nhưng thiếu lý do thì bị từ chối; hash manifest độc lập thứ tự
chèn dict; đổi 1 bệnh nhân đổi hash.

## 10. Kết quả test output/provenance guard (§6)

7/7 PASS: thư mục chưa tồn tại và thư mục rỗng đều được coi "fresh"; thư mục có file kết quả cũ
(`train_log.jsonl`) bị từ chối, **và nội dung thư mục đó không hề bị đụng tới** (xác nhận guard không
tự xoá gì); weight provenance ghi đúng SHA256 file giả, từ chối khi thiếu `weight_enum` hoặc thiếu file.

## 11. Đánh giá thiết kế v2 ở mức source (§7)

Bảng so sánh v1→v2 đầy đủ nằm trong docstring đầu file `run_hardened_verifier_v2.py`. Điểm cốt lõi:
mọi forward-only call trong khối critic-only giờ đi qua `preserved_eval_forward` (ép `.eval()` +
`inference_mode()`, khôi phục mode chính xác), bọc trong `GeneratorStateGuard` — assert buffer-hash +
parameter-version signature không đổi ngay sau mỗi khối, và assert full canonical hash tại các epoch
đã định trước (`PREDECLARED_FULL_HASH_EPOCHS`, không phải mọi batch — tránh chi phí runtime nặng nề).
`batch_policy` là tham số tường minh (`same_batch`/`fresh_batch`), `fresh_batch` cố tình **chưa nối**
với DataLoader thật (`_next_fresh_critic_batch` raise `NotImplementedError`) — đúng yêu cầu "không
triển khai/thực thi thật" cho nhánh chưa cần dùng ngay. **Cổng thực thi fail-closed**
(`_require_execution_manifest`) khiến `main()` luôn raise nếu không có file manifest đã được con người
duyệt tường minh (`human_approved: true`) — hiện tại **không có file như vậy trong repo**, nên
`main()` chắc chắn raise nếu ai đó gọi nhầm. File **không được import/instantiate** trong task này —
chỉ AST-parse.

## 12. Trạng thái triển khai bootstrap (§8)

`patient_graph_bootstrap.py` triển khai đúng công thức trọng số đã đặc tả (`weight=c_P` cho positive,
`weight=c_P*c_Q` cho negative), dùng `sample_weight` của `roc_auc_score` thay vì nhân bản dòng, phát
hiện one-class resample tường minh (raise mặc định, hoặc trả `None` tường minh nếu gọi
`raise_on_one_class=False` — không bao giờ âm thầm bỏ qua), RNG bootstrap độc lập hoàn toàn với seed
model/attacker. **`STATUS = 'UNVALIDATED'`** được giữ nguyên trong code — không đổi.

## 13. Simulation settings đã đông cứng trước khi chạy (§9)

```python
SETTINGS = {
    'monte_carlo_seed': 20260821,
    'n_simulated_datasets_per_scenario': 60,
    'n_bootstrap_replicates_per_dataset': 100,
    'coverage_target': 0.95, 'coverage_mc_band': 0.06,
    'type1_alpha': 0.05, 'type1_mc_band': 0.05,
    'balanced_scenario_invalid_resample_rate_max': 0.01,
    'non_inferiority_margin': 0.03,
}
```
6 kịch bản: `balanced_uniform_null`, `balanced_uniform_margin_null`, `balanced_high_degree_null`,
`imbalanced_uniform_null`, `sparse_correlated_null`, `alternative_improvement` — toàn bộ đã ghi cứng
trong file **trước khi chạy**, không sửa sau khi thấy kết quả.

## 14. Kết quả coverage/type-I-error — VÀ phát hiện lỗi hiệu chỉnh (calibration bug)

Chạy thật (`.venv/bin/python reproduction/statistics/simulate_bootstrap_coverage.py`, 69.1s CPU):

```
balanced_uniform_null          patient_cov=0.967 pair_cov=0.867 patient_type1=0.650 pair_type1=0.817
balanced_uniform_margin_null   patient_cov=0.500 pair_cov=0.250 patient_type1=0.650 pair_type1=0.817
balanced_high_degree_null      patient_cov=1.000 pair_cov=0.917 patient_type1=0.717 pair_type1=0.833
imbalanced_uniform_null        patient_cov=0.950 pair_cov=0.883 patient_type1=0.567 pair_type1=0.667
sparse_correlated_null         patient_cov=1.000 pair_cov=0.933 patient_type1=0.883 pair_type1=0.983
alternative_improvement        patient_cov=0.017 pair_cov=0.000 patient_type1=None  pair_type1=None
```

**Toàn bộ giá trị type-I error nằm ngoài dải cho phép đã đông cứng trước (≤0.10) — sai lệch rất lớn
(0.57–0.98).** Trước khi kết luận "bootstrap thất bại", tôi kiểm tra xem đây có phải lỗi ở chính bộ
tạo dữ liệu tổng hợp hay không (**không phải "tune lại phương pháp sau khi thấy kết quả"** — đây là
kiểm tra tính đúng đắn của công cụ đo, hợp lệ và bắt buộc trước khi tin bất kỳ số nào):

```
intended_delta=0.000  realized_delta=-0.0004  auc_b=0.9724  auc_c=0.9720
intended_delta=0.030  realized_delta= 0.0011  auc_b=0.9697  auc_c=0.9708
intended_delta=-0.060 realized_delta=-0.0006  auc_b=0.9685  auc_c=0.9679
```

**Xác nhận: bộ tạo dữ liệu tổng hợp (`_simulate_dataset`) bị lỗi hiệu chỉnh** — `arm_shift` gần như
không dịch chuyển AUC thực tế dù giá trị `true_delta` khai báo khác nhau (AUC nền đã bão hoà gần 0.97
do hàm sigmoid + tín hiệu signal=±1.0 đẩy điểm số về gần 0/1, khiến `arm_shift` nhỏ không đổi được thứ
hạng). Hệ quả: **mọi kịch bản thực tế đều có Δ thực ≈ 0 bất kể `true_delta` khai báo**, làm cho:
- Coverage kiểm tra sai mục tiêu (kiểm tra CI có chứa 0.03/-0.06 trong khi dữ liệu thực chỉ có Δ≈0).
- Type-I error đo trên nền dữ liệu đã bão hoà (AUC~0.97) không phản ánh đúng hành vi ở vùng AUC thực tế
  của dự án (~0.75-0.87, thấp hơn nhiều, ít bão hoà hơn).

## 15. Thất bại và giả định chưa giải quyết (nêu thẳng, không sửa ngầm)

- **Bản thân công cụ mô phỏng (`simulate_bootstrap_coverage.py`) có lỗi hiệu chỉnh, chưa được sửa
  trong task này** — theo đúng yêu cầu "không tune phương pháp sau khi thấy kết quả", tôi không sửa
  và chạy lại trong task này. Việc sửa bộ tạo dữ liệu (không phải sửa `patient_graph_bootstrap.py`)
  cần một vòng predeclare-rồi-chạy riêng, có phê duyệt riêng.
- **Do đó: không thể rút ra kết luận coverage/type-I hợp lệ nào cho phương pháp bootstrap từ lần chạy
  này.** Đây không phải "bootstrap PASS" và cũng không hẳn là "bootstrap FAIL" theo đúng nghĩa khoa
  học — là **INCONCLUSIVE do lỗi ở công cụ đo**, quy về đúng nhãn hiện tại: **`UNVALIDATED` (không
  đổi)**, không được nâng cấp thành phương pháp chính.
- `run_hardened_verifier_v2.py`'s `_next_fresh_critic_batch()` cố tình chưa nối với DataLoader thật —
  `fresh_batch` policy chưa khả thi để chạy, chỉ `same_batch` có đường thực thi đầy đủ (vẫn bị chặn bởi
  execution-manifest gate).
- `PREDECLARED_FULL_HASH_EPOCHS` trong v2 là **giá trị đề xuất**, chưa được con người duyệt chính thức
  — cần xác nhận trước khi dùng cho bất kỳ run thật nào.

## 16. Hành động GPU tương lai vẫn CHƯA được cấp phép

```
- Chạy run_hardened_verifier_v2.py dưới bất kỳ hình thức nào (bị chặn cứng bởi execution-manifest gate).
- Chạy determinism-fidelity diagnostic (G0.1 §2.3).
- Chạy G1 (Design B, G0.1A §4.3).
- Chạy lại phần đánh giá Stage A để lấy y_true/y_score.
- Sửa lại simulate_bootstrap_coverage.py và chạy lại (cần predeclare mới).
- Bất kỳ truy cập nào tới TEST.
```

---

```
G0.2 verdict: PASS

No GPU/CUDA workload was initialized.
No real model training or evaluation was run.
No real image or checkpoint was loaded.
No prohibited frozen-evaluation artifact was accessed or named.
No governed file was modified.
Direction B running process was not touched.

Awaiting separate human approval before any GPU diagnostic or scientific run.
```
