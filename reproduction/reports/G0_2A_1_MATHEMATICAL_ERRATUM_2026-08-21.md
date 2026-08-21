# G0.2A.1 MATHEMATICAL ERRATUM — Constant-Shift AUC Invariance

**Ngày:** 2026-08-21. Erratum thuần toán học — không sửa code, không chạy lại test/simulation, không
GPU/CUDA, không truy cập dữ liệu thật, không G1. Không sửa bất kỳ báo cáo trước nào. Chỉ tạo đúng 1
file: chính báo cáo này.

**An toàn:** không lệnh code nào được chạy để viết báo cáo này — toàn bộ là chứng minh giải tích trên
giấy, dựa trên đúng đoạn code đã trích dẫn nguyên văn ở
`G0_2A_AUDITABILITY_CLOSEOUT_2026-08-21.md` §B.1 (`simulate_bootstrap_coverage.py` dòng 93-105), không
đọc lại file.

---

## Sửa 1 — Nhiễu độc lập giữa 2 arm KHÔNG làm mất tính bất biến AUC của hằng số dịch

**`G0_2A_AUDITABILITY_CLOSEOUT_2026-08-21.md` §B.1 đã sai** khi nói tính bất biến thứ hạng "chỉ áp
dụng một phần" và cần `idio1` dùng chung giữa 2 arm mới có bất biến tuyệt đối. Đây là hiểu sai bản
chất toán học. Sửa lại cho đúng:

**Chứng minh:** đặt `z_i = signal_i + shared_i + epsilon_candidate_i` — đây là giá trị **cố định**,
không phụ thuộc `c` (với `c = true_delta × 2.5`, chính là `arm_shift`). Điểm số candidate:
```
candidate_score_i(c) = sigmoid(z_i + c)
```
Với 2 quan sát bất kỳ `i, j` (bất kể cùng arm hay khác arm, bất kể `epsilon_candidate` có được rút mẫu
độc lập hay không — điều này **không xuất hiện** trong so sánh dưới đây):
```
candidate_score_i(c) > candidate_score_j(c)
⟺ z_i + c > z_j + c        (vì sigmoid tăng ngặt, đơn điệu)
⟺ z_i > z_j                 (c triệt tiêu ở cả 2 vế)
```
Điều kiện `z_i > z_j` **hoàn toàn không phụ thuộc `c`**. Vì AUC (dạng Mann-Whitney U) được định nghĩa
thuần tuý từ các so sánh thứ hạng từng cặp (positive, negative):
```
AUC(c) = (1/(n_pos·n_neg)) · Σ [ 1{score_pos > score_neg} + 0.5·1{score_pos = score_neg} ]
```
và mọi số hạng trong tổng này **không đổi khi `c` đổi** (vì điều kiện so sánh chỉ phụ thuộc `z_i, z_j`,
không phụ thuộc `c`), nên:
```
AUC_candidate(c) = AUC_candidate(0)     với MỌI hằng số c hữu hạn — ĐÚNG TUYỆT ĐỐI, không phải "gần đúng".
```
**Kết quả này không cần giả định `epsilon_baseline` và `epsilon_candidate` dùng chung 1 lần rút mẫu.**
Đây là thuộc tính của **riêng arm candidate**: cộng cùng 1 hằng số vào **mọi** quan sát của arm đó rồi
biến đổi đơn điệu, giữ nguyên **mọi** so sánh thứ hạng nội bộ arm đó — không liên quan gì đến việc arm
kia (baseline) có chia sẻ nhiễu hay không.

**Diễn giải đúng, sửa lại hoàn toàn:**
- Nhiễu candidate được rút mẫu độc lập (`idio1` khác giữa 2 lần gọi `pair_score()`) **có thể** khiến
  `AUC_candidate` khác `AUC_baseline` một cách **ngẫu nhiên** (do 2 arm có bản chất phân phối điểm số
  khác nhau ngẫu nhiên, không liên quan `arm_shift`).
- Nhưng `arm_shift` (`c`) **có tác động đúng bằng 0 lên `AUC_candidate`** — không phải "quá nhỏ", không
  phải "bị nhiễu lấn át" như báo cáo trước diễn giải sai. Tác động là **0 tuyệt đối**, không phải một
  con số nhỏ do biên độ.
- Các Δ nhỏ quan sát được (-0.0004, +0.0011, -0.0006 trong `G0_2_IMPLEMENTATION_REPORT_2026-08-21.md`
  §14) **hoàn toàn đến từ các lần rút mẫu nhiễu độc lập khác nhau giữa baseline và candidate**, không
  đến từ hằng số dịch bị "swamped" — hằng số dịch chưa từng có cơ hội ảnh hưởng tới AUC ngay từ đầu.
- **Tăng biên độ hằng số dịch không thể sửa được bộ tạo dữ liệu (DGP)** — dù `c` lớn tới đâu (trừ
  trường hợp bão hoà số học ở sigmoid gây trùng lặp giá trị làm tròn, xem ghi chú dưới), `AUC_candidate`
  vẫn không đổi.

**Ghi chú về "trùng lặp do bão hoà số học" (numerical ties):** nếu `z_i + c` đủ lớn để `sigmoid(z_i+c)`
làm tròn về đúng `1.0` (hoặc đủ nhỏ để làm tròn về đúng `0.0`) trong floating-point, 2 quan sát có
`z_i ≠ z_j` khác nhau nhưng đều rơi vào vùng bão hoà có thể bị làm tròn thành **cùng 1 giá trị điểm số**
— tạo ra 1 cặp hoà (tie) không tồn tại trong số học chính xác. Đây là hiệu ứng số học bậc hai, không
phải cơ chế chính, và không đổi kết luận: tác động thật của `c` lên AUC là 0 trong số học chính xác,
gần-0 (do vài tie số học hiếm) trong thực thi floating-point.

## Sửa 2 — Không có ước lượng type-I error hợp lệ nào trong simulation v1

Kịch bản khai báo `true_delta = +0.03` **không hề tạo ra Δ AUC thực tế gần +0.03** — vì theo Sửa 1,
hằng số dịch không có tác động lên AUC candidate, Δ thực tế luôn dao động quanh 0 (do nhiễu độc lập
ngẫu nhiên giữa 2 arm), bất kể `true_delta` khai báo là gì.

**Do đó, tỷ lệ bác bỏ H0 đã báo cáo cho kịch bản này KHÔNG được mô tả là:**
- type-I error hợp lệ;
- type-I error bị "thổi phồng" (inflated);
- hay 1 vấn đề type-I error thật nhưng "bị confound bởi lỗi hiệu chỉnh" (cách diễn đạt này trong
  `G0_2A_AUDITABILITY_CLOSEOUT_2026-08-21.md` §B.2 **cũng cần sửa** — nó vẫn ngầm giả định có 1 giá trị
  type-I error "thật" đâu đó bị nhiễu che khuất; thực tế không có phép đo type-I error nào diễn ra,
  vì điều kiện null `delta=+0.03` chưa từng thực sự được tạo ra trong dữ liệu).

Tỷ lệ bác bỏ đó là kết quả đo tại **delta thực tế ≈ 0** — gần với ý nghĩa "power của non-inferiority
dưới 1 DGP bị sai thông số kỹ thuật" (misspecified DGP) hơn là type-I error.

Tương tự, các kịch bản `true_delta=0` khác cũng bị gắn nhãn sai là "type-I scenario" (đã nêu ở
G0.2A §B.2) — thực chất là các kịch bản power, và **cũng chịu chung hạn chế của DGP này** (dù về mặt
khái niệm đúng là power-scenario, giá trị đo được không đáng tin vì DGP không tách bạch được các mức
`true_delta` khác nhau — mọi kịch bản, dù khai báo delta gì, đều hội tụ về cùng 1 delta thực ≈ 0).

**Kết luận bắt buộc:**
```
simulation_v1_valid_type1_estimate: NONE
simulation_v1_valid_power_estimate: NONE
simulation_v1_valid_coverage_conclusion: NONE
```

## Sửa 3 — DGP tương lai (G0.2B) phải đổi độ phân tách lớp, không phải đổi intercept điểm số

Không triển khai, không chạy — chỉ ghi nhận yêu cầu thiết kế: một DGP hợp lệ cho candidate phải thay
đổi **chính các đại lượng quyết định AUC**, ví dụ:
```
class-conditional separation (độ phân tách trung bình giữa lớp dương/âm)
signal coefficient (hệ số nhân của "signal")
noise variance (phương sai nhiễu)
signal-to-noise ratio
```
Phác thảo thiết kế tương lai (chỉ để tham khảo, không triển khai):
```
raw_baseline  = separation_B · label_sign + patient_effect + noise_B
raw_candidate = separation_C · label_sign + patient_effect + noise_C
```
với `separation_C` được hiệu chỉnh (calibrate) để tạo ra đúng AUC candidate mục tiêu — **không phải**
cộng thêm 1 hằng số vào điểm số sau cùng.

Hằng số dịch (constant shift) nên được giữ lại — nhưng chuyển vai trò thành **1 unit test kiểm chứng
âm (negative-control)**: chứng minh tường minh rằng AUC không đổi khi chỉ cộng hằng số — đúng là điều
Sửa 1 vừa chứng minh, nên biến nó thành 1 bài test xác nhận thay vì (nhầm) dùng làm cơ chế tạo hiệu ứng.

**Task này không triển khai hay chạy thiết kế trên.**

---

## Verdict bắt buộc

```
G0_2A_previous_DGP_mechanism_interpretation: CORRECTED
constant_shift_effect_on_AUC: EXACTLY_ZERO_EXCEPT_NUMERICAL_TIES
independent_noise_effect: RANDOM_BETWEEN_ARM_AUC_DIFFERENCE_ONLY
simulation_v1_valid_type1_estimate: NONE
simulation_v1_valid_power_estimate: NONE
simulation_v1_valid_coverage_conclusion: NONE
patient_graph_bootstrap: UNVALIDATED
overall_G0_2: PARTIAL_PASS_BLOCKED
GPU_AUTHORIZATION: NONE
```

Không file nào đã tồn tại bị sửa. Không thí nghiệm/test/simulation nào được chạy trong task này —
toàn bộ là chứng minh giải tích dựa trên đoạn code đã trích dẫn sẵn từ báo cáo trước. Tiến trình
Direction B (PID quan sát trước khi viết báo cáo này) không bị đụng tới. Dừng lại, chờ external review.
