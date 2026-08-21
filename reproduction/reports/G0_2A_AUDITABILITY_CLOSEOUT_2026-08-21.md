# G0.2A AUDITABILITY CLOSEOUT — Source Review Bundle Only

**Ngày:** 2026-08-21. Chỉ audit source + đóng gói bundle — không sửa implementation, không chạy lại
simulator, không GPU, không khởi động G1/thí nghiệm khoa học nào.

**An toàn:** `ps aux | grep run_hardened_verifier` xác nhận PID 1875303 vẫn chạy trước/trong/sau task
này, không bị đụng. Branch `research/method-restart`, HEAD `34bca1f74662275ced0418ea183a3d9d5ef81f88`,
không đổi.

---

## Task A — Review bundle

### A.1 `tar -tzf` output đầy đủ

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
reproduction/reports/G0_2_IMPLEMENTATION_REPORT_2026-08-21.md
```
Đúng 14 file (13 file implementation/test + 1 báo cáo G0.2), không có `__pycache__`, `.pyc`, dataset,
checkpoint, log, hay artifact nào khác.

### A.2 Đường dẫn, kích thước byte, SHA256 từng file

| File | Bytes | SHA256 |
|---|---:|---|
| `reproduction/protocol_v2/__init__.py` | 271 | `5a2bf9161ea3a4a17c3a61edcae8d9c6ed0ef0c7adbb1fa1ebc6429a66b94be5` |
| `reproduction/protocol_v2/state_invariants.py` | 3254 | `7c0b8eeb0b005f41da651cc776442e24989dfc3680b30c7c27cf2480edde20b5` |
| `reproduction/protocol_v2/deterministic_loader.py` | 3328 | `70a4487a3e20b61fe1f44e1f50697e6cc924fd785eda2a46b4a3fe963e9f460e` |
| `reproduction/protocol_v2/role_manifest.py` | 4295 | `909bc0e0a7aaa858b15a3a00df0e0aab2f84e8dbb0a3a58f9eaf535a51bfca24` |
| `reproduction/protocol_v2/output_freshness.py` | 2319 | `1438f37dd55ca5b65c9234c05015622f869f7e8d15f5793030763c90c83b9b9b` |
| `reproduction/protocol_v2/weight_provenance.py` | 2555 | `a31d185916744ffa90aca23d0afa4c3fa79c6cc98001ab8a0b38d4f0d2f0f339` |
| `reproduction/statistics/__init__.py` | 149 | `c711fc947166cc9a7e8f6c3caad40bdb2aab9afaedd13b142b0b107db7782991` |
| `reproduction/statistics/patient_graph_bootstrap.py` | 4987 | `be25735e5d067e0d3fb2d97a4042c10c4ed9b60b1f1450313f7b4e31cba9bbcd` |
| `reproduction/statistics/simulate_bootstrap_coverage.py` | 11294 | `eb020b2b736ba5a441f25e8c03bbf91337b47e7d25679c240f87ba9734fa3ce2` |
| `reproduction/method_dev/run_hardened_verifier_v2.py` | 21835 | `f096b2f5e4401d84749ab70d5ffe4bb2eb673648d9f983ab9cbf897a52da5497` |
| `reproduction/tests/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `reproduction/tests/test_protocol_v2.py` | 13146 | `06dc1e476aca0cf53a73b44bb72c6d292ad4a89b1c5e2f189b72ab31b0f0dd9a` |
| `reproduction/tests/test_patient_graph_bootstrap.py` | 5995 | `7031d74774262d30eec55b4a88df3cedee85655c9bc5a2410e0d1dc637c17ce2` |
| `reproduction/reports/G0_2_IMPLEMENTATION_REPORT_2026-08-21.md` | 13536 | `4fa9d47fb55565f7cd418948f432107cd1768d44a0cc3b449d2c83c310922a4e` |

### A.3 SHA256 của archive cuối cùng

```
59a9b70cde7b49aca3867ad3cce733dd37b8c9f5bfeda24ed62b08bd2987a764  reproduction/reviews/G0_2_REVIEW_BUNDLE_2026-08-21.tar.gz
```
(26.750 bytes, cũng ghi trong file `.sha256` đi kèm)

### A.4 `git check-ignore -v` cho từng file G0.2

Tất cả 14 file đều khớp đúng 1 rule duy nhất:
```
.gitignore:17:reproduction/    <mỗi file trên>
```

### A.5 Đính chính bắt buộc

> `git diff --check` không kiểm tra file bị ignore và untracked. Do đó, việc nó trả về exit 0 trước
> đây **không thể** dùng làm bằng chứng rằng các file G0.2 mới này sạch hay đúng — nó chỉ xác nhận
> không có whitespace-conflict-marker trên các file **đã tracked** (không file G0.2 nào thuộc diện
> này). `.gitignore` không bị sửa để lộ các file này ra.

---

## Task B — Source audit (không sửa code)

### B.1 Bộ tạo AUC tổng hợp (`simulate_bootstrap_coverage.py`)

**Trace chính xác `arm_shift`:**
```python
# dòng 93-98
def pair_score(p1, p2, arm_shift, arm_noise_scale):
    shared = patient_correlation * (patient_trait[p1] + patient_trait[p2]) / 2
    idio1 = (1 - patient_correlation) * rng.normal(0, 1, size=len(p1))
    signal = np.where(p1 == p2, 1.0, -1.0)
    raw = signal + shared + idio1 * arm_noise_scale + arm_shift
    return 1 / (1 + np.exp(-raw))
# dòng 104-105
y_score_baseline = pair_score(patient1, patient2, arm_shift=0.0, arm_noise_scale=1.0)
y_score_candidate = pair_score(patient1, patient2, arm_shift=true_delta * 2.5, arm_noise_scale=1.0)
```

**Không hoàn toàn đúng dạng `candidate_score = sigmoid(base_score + constant)` như giả thuyết đặt ra —
có 1 khác biệt quan trọng cần nêu chính xác:** `idio1` được rút mẫu **độc lập** ở mỗi lần gọi
`pair_score()` (vì `rng.normal()` tiêu thụ trạng thái RNG mới ở lần gọi thứ 2 cho candidate) — nghĩa là
baseline và candidate **không dùng chung 1 điểm số gốc** rồi cộng hằng số, mà dùng chung `signal` +
`shared` (tương quan theo bệnh nhân) nhưng nhiễu riêng (`idio1`) độc lập cho mỗi arm.

**Vì vậy có 2 cơ chế cùng góp phần vào hiện tượng "Δ thực ≈ 0 bất kể true_delta", cần tách bạch:**
1. **Bất biến thứ hạng (rank invariance)** — đúng như giả thuyết, nhưng chỉ áp dụng **một phần**: nếu
   `idio1` được dùng chung (không redraw), thì `raw_candidate = raw_baseline + arm_shift` là phép cộng
   hằng số đồng loạt lên MỌI quan sát, và vì sigmoid là đơn điệu tăng ngặt, **AUC(sigmoid(raw+c)) ==
   AUC(sigmoid(raw)) tuyệt đối, đúng bằng nhau (không phải gần bằng)** — vì AUC = P(score dương >
   score âm), và phép biến đổi đơn điệu áp dụng đồng loạt cho mọi quan sát giữ nguyên **mọi** so sánh
   thứ hạng từng cặp. Đây là 1 sự thật toán học, không phụ thuộc code cụ thể.
2. **Nhiễu độc lập lấn át hằng số dịch (magnitude swamping)** — vì `idio1` **có** redraw độc lập trong
   code thực tế, AUC không bất biến tuyệt đối (bằng chứng: Δ thực đo được là -0.0004/+0.0011/-0.0006,
   khác 0 dù rất nhỏ — không phải đúng 0 tuyệt đối như cơ chế (1) sẽ cho). Nhưng biên độ `arm_shift`
   (tối đa `0.06×2.5=0.15`) quá nhỏ so với biên độ nhiễu (`idio1` std≈1, `signal`=±1.0) — hằng số dịch
   gần như bị nhiễu ngẫu nhiên độc lập nuốt mất, không đủ mạnh để dịch chuyển AUC đo được có ý nghĩa.

**Đính chính đúng theo yêu cầu:** lỗi thiết kế **chính** không đơn thuần là "sigmoid saturation" như
diễn giải trước đó ngụ ý — mà là **tổ hợp của (a) hằng số dịch quá nhỏ so với nhiễu, và (b) nếu từng
phần chia sẻ nhiễu chung, sẽ có thêm hiệu ứng bất biến thứ hạng tuyệt đối do biến đổi đơn điệu**. Cả
hai đều dẫn tới cùng 1 hệ quả quan sát được (Δ thực ≈ 0), nhưng là 2 cơ chế toán học khác nhau, và code
thực tế khớp với cơ chế (2) nhiều hơn (do có redraw), không phải cơ chế (1) thuần tuý. **Không sửa bộ
tạo dữ liệu trong task này.**

### B.2 Quy ước dấu và giả thuyết thống kê

**Quy ước dấu trong code:** `patient_graph_bootstrap.py` dòng `deltas.append(auc_c - auc_b)` — đúng
`delta = AUC_candidate − AUC_baseline`, khớp quy ước yêu cầu.

**Giả thuyết đúng cho non-inferiority (margin m=0.03):**
```
H0: delta >= +0.03
H1: delta <  +0.03
```
Ranh giới type-I error đúng đắn: **delta = +0.03**. Tại `delta = 0`, việc bác bỏ H0 là **power** (khả
năng phát hiện đúng), **không phải** type-I error.

**Phát hiện lỗi nhãn cụ thể trong code (`simulate_bootstrap_coverage.py`):**
```python
# dòng 182
is_null_scenario = abs(true_delta - settings['non_inferiority_margin']) < 1e-9 or true_delta == 0.0
# dòng 190, 196
'type1_error_if_null_scenario': (n_reject_h0_patient / n) if is_null_scenario else None,
```
**Xác nhận đúng như nghi ngờ:** dòng 182 gộp cả `true_delta==0.0` LẪN `true_delta==margin` vào chung 1
cờ `is_null_scenario`, rồi cả 2 trường hợp đều bị gắn nhãn `type1_error_if_null_scenario` ở dòng
190/196. Đây là **lỗi nhãn thật, xác nhận bằng chính source, không phải suy đoán**.

**Hệ quả cho các con số đã báo cáo ở `G0_2_IMPLEMENTATION_REPORT_2026-08-21.md` §14 — cần đọc lại
đúng:**
- Chỉ **`balanced_uniform_margin_null`** (true_delta=+0.03, đúng ranh giới H0) mới thực sự đo type-I
  error. Giá trị đã báo cáo trước đó cho kịch bản này: **patient_type1=0.650** — vẫn là vấn đề thật
  (vượt xa dải cho phép ≤0.10), nhưng bị confound bởi lỗi hiệu chỉnh ở §B.1 (Δ thực không đạt 0.03).
- 4 kịch bản còn lại có `true_delta=0.0` (`balanced_uniform_null`, `balanced_high_degree_null`,
  `imbalanced_uniform_null`, `sparse_correlated_null`) — **các giá trị "type1_error" 0.567–0.883 đã báo
  cáo trước đó thực chất là POWER (đo tại delta=0, sâu trong vùng H1), không phải type-I error.** Việc
  gọi chúng là "type-I error" trong báo cáo G0.2 trước là **sai nhãn, cần sửa nhận thức**, dù không sửa
  file đó. Giá trị cao ở các kịch bản này (nếu số liệu hiệu chỉnh đúng) sẽ là **tin tốt** (power cao),
  không phải bằng chứng lỗi phương pháp — nhưng vì lỗi hiệu chỉnh ở §B.1 vẫn tồn tại, ngay cả cách đọc
  "power" này cũng chưa thể tin cậy từ lần chạy đó.

**Không tự ý diễn giải lại kết quả cũ thành "đã sửa xong"** — chỉ nêu đúng những gì code làm và ý nghĩa
thống kê chính xác của từng con số.

### B.3 Triển khai bootstrap

1. **Loại khoảng tin cậy:** percentile bootstrap thuần (`np.percentile(deltas, [2.5, 97.5])` trong
   `simulate_bootstrap_coverage.py`; bản thân `patient_graph_bootstrap.py` không tự tính CI, chỉ trả
   list delta thô, để caller tự tính). Không phải BCa, basic, hay studentized.
2. **Quyết định non-inferiority:** một phía, `upper95 = percentile(deltas, 95); reject H0 nếu upper95 <
   margin` (dòng 168-169, 177-178 `simulate_bootstrap_coverage.py`).
3. **Cùng resample cho baseline/candidate:** đúng — `patient_graph_bootstrap_paired()` tính đúng 1
   `draw_counts`/`weights` mỗi replicate, dùng chung cho cả `weighted_auc(...,y_score_baseline,...)`
   và `weighted_auc(...,y_score_candidate,...)`.
4. **Công thức trọng số:** đúng khớp `weight=c_P` (positive), `weight=c_P*c_Q` (negative) —
   `pair_weights_for_draw()`, đã kiểm chứng bằng test so khớp với row-expansion tường minh (14/14 PASS,
   §C).
5. **`sample_weight` dùng đúng:** `roc_auc_score(yt, ys, sample_weight=w)` sau khi lọc `weights>0` —
   đúng tham số sklearn, không tự tính AUC thủ công.
6. **Xử lý one-class resample:** `weighted_auc(..., raise_on_one_class=True)` mặc định **raise**
   `OneClassResampleError`; `patient_graph_bootstrap_paired()` gọi với `raise_on_one_class=False` nội
   bộ, nhận `None` tường minh, **đếm vào `n_one_class_invalid`, không thêm vào `deltas`** — không bao
   giờ âm thầm coi là 0 hay giá trị hợp lệ khác.
7. **Invalid resample:** được đếm tường minh (`n_one_class_invalid`), trả về cho caller cùng với
   `n_valid` — không bị "silently discarded" theo nghĩa mất dấu vết, nhưng cũng không nằm trong danh
   sách `deltas` cuối cùng (đúng thiết kế: loại khỏi tính CI, nhưng giữ đếm để báo cáo tỷ lệ).
8. **RNG bootstrap độc lập:** `rng = np.random.default_rng(bootstrap_seed)` — tham số `bootstrap_seed`
   tách biệt hoàn toàn khỏi mọi `attacker_seed`/`generator_seed` trong toàn bộ codebase đã kiểm tra.

**Không nâng cấp trạng thái bootstrap dù source khớp đúng đặc tả** — `STATUS = 'UNVALIDATED'` giữ
nguyên trong code, đúng yêu cầu.

### B.4 Runner v2 và bảo toàn state

1. Mọi forward-only call của generator trong khối critic-only đều đi qua `preserved_eval_forward`
   (`_critic_only_fake()` gọi `with preserved_eval_forward(self.generator): return
   self.anonymize_tensor(inputs1)`) — dùng `.eval()` + `torch.inference_mode()`, khôi phục đúng mode
   gốc (đã kiểm chứng thực nghiệm ở §C, 12/12 test state-invariant PASS).
2. Bất biến parameter/buffer được kiểm tra qua `GeneratorStateGuard.verify_unchanged()` — so
   `buffer_only_hash`, `parameter_version_signature`, và (tại epoch định trước) full canonical hash.
3. `PREDECLARED_FULL_HASH_EPOCHS` **vẫn là hằng số chưa được con người phê duyệt** — đúng như đã nêu ở
   `G0_2_IMPLEMENTATION_REPORT_2026-08-21.md` §15, xác nhận lại không đổi.
4. `_next_fresh_critic_batch()` vẫn `raise NotImplementedError(...)` — `fresh_batch` policy chưa hoàn
   thiện, chưa có đường thực thi.
5. **Runner chưa bao giờ được import hay chạy trong G0.2/G0.2A** — xác nhận qua: (a) mọi lần kiểm tra
   chỉ dùng `ast.parse()`, không `import`; (b) không file test nào (`test_protocol_v2.py`,
   `test_patient_graph_bootstrap.py`) import `run_hardened_verifier_v2`; (c) không lệnh nào trong lịch
   sử phiên làm việc gọi `main()` của file này.

### B.5 Bảo mật execution-manifest

Cổng hiện tại (`_require_execution_manifest`, `run_hardened_verifier_v2.py`):
```python
if not manifest.get('human_approved') is True:
    raise RuntimeError(...)
```
**Xác nhận đúng lo ngại:** chỉ kiểm tra 1 trường boolean `human_approved: true` trong JSON — **không
đủ** làm khoá thực thi nghiên cứu thật. Một manifest tương lai cần ràng buộc tối thiểu: SHA256 chính
xác của runner, SHA256 các module helper, HEAD repo, branch, đường dẫn config + SHA256, định danh +
hash checkpoint/generator, seed, tỷ lệ update verifier, batch policy, lệnh đã duyệt, thư mục output,
hash role-manifest kỳ vọng, và định danh/bản ghi người đã phê duyệt. **Không triển khai cổng mạnh hơn
trong task này** — chỉ ghi nhận thiếu sót.

---

## C. Kiểm chứng CPU-only cho phép (rerun test đã khai báo trước)

```bash
export CUDA_VISIBLE_DEVICES=""
.venv/bin/python reproduction/tests/test_protocol_v2.py             # exit 0, 38/38 PASS
.venv/bin/python reproduction/tests/test_patient_graph_bootstrap.py  # exit 0, 14/14 PASS
```
`test_protocol_v2.py` dòng đầu tự assert `not torch.cuda.is_initialized()` — PASS. Không file nào
trong 2 test này import/instantiate `run_hardened_verifier_v2`, không load dataset/checkpoint thật.

**Kết quả test PASS không được dùng làm bằng chứng phương pháp thống kê đúng** — chỉ xác nhận code
chạy không lỗi theo đúng input tổng hợp đã viết, không xác nhận tính hợp lệ khoa học của
patient-graph bootstrap (vẫn `UNVALIDATED`, xem §B.3, §B.1-B.2).

---

## D. Kết luận bắt buộc

1. Protocol scaffolding G0.2 có thể đã pass unit test đã báo cáo, nhưng **implementation chưa qua
   independent external source review** — task này là bước đầu tiên hướng tới đó (bundle đã đóng gói),
   chưa phải bản thân review độc lập.
2. Lần simulation đầu tiên **không thể** xác nhận hay bác bỏ patient-graph bootstrap vì bộ tạo dữ liệu
   (data-generating process) của nó không tạo ra đúng Δ AUC đã khai báo (§B.1).
3. Lỗi cốt lõi của simulator không đơn thuần là "sigmoid saturation" — code thực tế có redraw nhiễu độc
   lập mỗi arm, nên là tổ hợp giữa hằng số dịch quá nhỏ so với nhiễu VÀ (nếu nhiễu được chia sẻ) bất
   biến thứ hạng tuyệt đối do biến đổi đơn điệu (§B.1).
4. Type-I error của non-inferiority phải đo tại `delta=+0.03`; `delta=0` đo power, không phải type-I —
   xác nhận code hiện tại **gộp nhầm** 2 khái niệm này vào chung 1 nhãn (§B.2, lỗi thật, đã trích dòng).
5. 60 dataset mô phỏng × 100 bootstrap replicate là **không đủ** cho việc xác nhận thống kê cuối cùng
   của phương pháp — chỉ đủ cho sàng lọc sơ bộ.
6. Bất định bootstrap cấp-bệnh-nhân **không thay thế** được lặp lại theo generator-seed — không thể tự
   nó làm bằng chứng cho tuyên bố cấp-phương-pháp để công bố.
7. **Không có chẩn đoán GPU hay chạy khoa học nào được cấp phép** trong task này.

---

## E. Verdict theo từng thành phần (không gộp thành 1 PASS duy nhất)

```
protocol_tests: REPORTED_PASS
independent_external_source_review: PENDING
runner_v2_execution_status: NOT_VALIDATED
fresh_batch_status: NOT_IMPLEMENTED
patient_graph_bootstrap: UNVALIDATED
simulation_v1: INVALID_DGP / INCONCLUSIVE
overall_G0_2: PARTIAL_PASS_BLOCKED
GPU_AUTHORIZATION: NONE
```

```
No GPU/CUDA workload was initialized.
No real model training or evaluation was run.
No real image, dataset, or checkpoint was loaded.
No prohibited frozen-evaluation artifact was named or accessed.
No existing report, governed file, or implementation file was modified.
The running Direction B process was not touched.
No commit or push was performed.
```
