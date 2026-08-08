# PLAN — Vượt sàn 0.577: vì sao anonymization đối kháng chững lại, và cách sửa

## 0. Luận điểm trung tâm

PriCheXy-Net học một trường biến dạng **tất định**: `F = UNet(x)`, warp trơn (Gauss σ=2), biên độ
chặn bởi µ. Một ánh xạ như vậy gần như song ánh, nên về mặt lý thuyết thông tin nó **không xoá được
identity** — `I(F(X); ID) ≈ I(X; ID)`. Nó chỉ đổi hệ mã. Attacker train lại trên dữ liệu anonymized
chỉ cần học hệ mã mới.

Đó là lý do Re-ID chững ở ~0.58 chứ không xuống 0.50, và bằng chứng nằm ngay trong loss curve của
chính ta (mục 2). Nhìn lại Table 1 của paper gốc, quy luật rất rõ:

| Phương pháp | Ngẫu nhiên? | Ver. AUC ↓ | Class. AUC ↑ |
|---|---|---|---|
| DP-Pix (b=1…8) | **Có** (Laplace) | **50.0 – 52.5** | 50.0 – 52.9 ✗ |
| Privacy-Net | Không | 49.8 ± 2.2 | 57.5 ✗ |
| PriCheXy-Net µ=0.01 | **Không** | 57.7 ± 4.0 | **76.2** ✓ |
| Ảnh gốc | — | 81.8 ± 0.6 | 80.5 |

Chỉ phương pháp **có ngẫu nhiên** mới chạm 50. Ô trống — *ngẫu nhiên **và** giữ utility* — là chỗ ta
nhắm tới.

### ⛔ Luận điểm trên ĐÃ BỊ BÁC BỎ bằng thực nghiệm (xem D4)

Control λ=1 (warp **hoàn toàn ngẫu nhiên**, cùng µ) cho Re-ID **0.7942 ± 0.032** — nằm trong một std của
ảnh gốc (0.8015). Ngẫu nhiên ở biên độ µ=0.01 mang lại **gần như không có privacy nào**. Trong khi biến
dạng *học được* cho 0.6038.

**Luận điểm thay thế (đã có bằng chứng):**

> Privacy của PriCheXy-Net mang **bản chất đối kháng**, không phải lý thuyết thông tin. Nó khai thác biên
> quyết định của attacker — giống một adversarial perturbation — chứ không xoá thông tin sinh trắc khỏi ảnh.

Điều này giải thích gọn toàn bộ Table 1 của paper gốc:
- Attacker train lại lấy lại được nhiều (0.58–0.60, không phải 0.50) → **thông tin vẫn còn nguyên trong ảnh**.
- Nhiễu không nhắm đích cùng biên độ → 0 privacy (D4).
- DP-Pix phải **phá huỷ** ảnh (b=8) mới chạm 0.50 → nhiễu không nhắm đích cần biên độ khổng lồ.

**Hệ quả cho hướng đi:** ở một ngân sách biến dạng µ cố định, **nhắm đích là tất cả, entropy vô giá trị**.
Đường tới privacy tốt hơn là **nhắm tốt hơn** → ưu tiên C2 (phân bổ theo giải phẫu) và C3 (ensemble +
restart). C1 bị loại khỏi phần đóng góp, giữ lại làm một dòng ablation.

---

## ⛔ 0.5. CẢNH BÁO: mọi kết quả bên dưới đến từ một generator BỊ HUẤN LUYỆN LỖI

Phát hiện ngày 2026-08-06 khi rà soát lại code so với repo gốc.

"Gradient accumulation" thêm vào ở H2 bị hỏng:

```python
optimizer_g.zero_grad()                       # <-- xoá gradient MỖI iteration
(total_loss / accumulation_steps).backward()
if (i + 1) % accumulation_steps == 0:
    optimizer_g.step()
```

`zero_grad()` ở đầu vòng lặp xoá sạch gradient vừa tích luỹ. Hậu quả: mỗi `step()` chỉ thấy gradient của
**một** batch chia cho 4, và generator được cập nhật **ít hơn 4 lần**. Kiểm chứng số: gradient của code cũ
không khớp gradient đúng (lệch dấu ở 2/4 thành phần, độ lớn 0.315×).

Code gốc (`git show HEAD:utils/utils.py:353`) không có gì như vậy — nó `zero_grad / backward / step` mỗi
iteration, đúng chuẩn.

**Điều này giải thích chính xác khoảng cách repro:** generator bị dưới-huấn-luyện → biến dạng ít hơn →
Re-ID **cao hơn** paper (0.604 vs 0.577) **và** utility **cao hơn** paper (0.770 vs 0.762). Đúng cả hai chiều.

**Đã sửa** (`utils/utils.py`, bỏ `zero_grad()` đầu vòng lặp; `accumulation_steps` mặc định = 1 để trung
thành với paper).

### Kết luận nào còn đứng vững, kết luận nào phải rút lại

| | Trạng thái |
|---|---|
| **D4** (warp ngẫu nhiên = 0 privacy) | ✅ **Vẫn đúng** — λ=1 bỏ qua generator hoàn toàn, không phụ thuộc huấn luyện |
| **D3** phân rã BCE dương/âm (+0.415 / −0.016, pha loãng 12.3×) | ✅ **Vẫn đúng** — tính từ prediction trên test set, là tính chất của dữ liệu + loss |
| **D3** "ac_loss phẳng suốt 250 epoch" | ⚠️ **Nhiễu loạn** — có thể chỉ vì generator gần như không được train |
| **D5** (ensemble/restart không giúp) | ⚠️ So sánh *tương đối* còn giá trị (cùng lỗi), nhưng kết luận tuyệt đối không vững |
| **D6** (C2 làm sụp utility) | ⚠️ Như D5 |
| **"µ là ràng buộc chặt"** | ❌ **PHẢI RÚT LẠI** — ta chưa từng thực sự tối ưu generator |

**Việc bắt buộc:** chạy lại baseline + các ablation chính với code đã sửa trước khi kết luận bất cứ điều gì
về phương pháp.

## 1. Kết quả đã đo (protocol 10 seeds)

| Run | Checkpoint | Re-ID AUC (10 seeds) | Class. AUC | Vị trí |
|---|---|---|---|---|
| Paper | — | 0.577 ± 0.040 | 0.762 | tham chiếu |
| Ảnh gốc (paper) | — | 0.818 ± 0.006 | 0.805 | upper bound |
| Ảnh gốc (ours) | — | **0.8015 ± 0.0270** | **0.8050** | `archive/retrain_snn_runs_baseline_none/`, `chexnet/results/test_baseline_none/` |
| run_1 — repro gốc | lowest_total | **0.604 ± 0.082** | 0.770 | `archive/retrain_snn_runs_total/` |
| run_1 — repro gốc | lowest_ver | 0.641 ± 0.058 | — | `archive/retrain_snn_runs_verloss/` |
| run_2 — H1 fix | lowest_total | 0.622 ± 0.052 | 0.755 | `archive/retrain_snn_runs_run2_total/` |
| run_3 — entropy loss | lowest_total | 0.706 ± 0.050 | 0.786 | `archive/retrain_snn_runs_run3_total/` |
| run_4 — ensemble+restart (C3) | lowest_total (ep54) | **0.6060 ± 0.0478** | **0.7617** | `archive/retrain_snn_runs_run4_ensemble/` |
| **control λ=1 — warp ngẫu nhiên** | — | **0.8174 ± 0.0287** (n=10) | *chưa đo* | `archive/retrain_snn_runs_randomwarp/` |
| **C2 — budget map** | lowest_total (ep58) | **0.7600 ± 0.0256** | **0.7090** | `archive/train_prichexy_net_c2_budgetmap/`, `chexnet/results/test_c2/` |

Repro đạt: 0.604 nằm trong 1 std của 0.577. Ảnh gốc của ta 0.8015 khớp tốt với 0.818 của paper.

---

## 2. Chẩn đoán — ba phát hiện đã kiểm chứng từ loss curve run_1

Nguồn: `archive/train_prichexy_net_run_1/loss_dict.pkl`. Ba phát hiện này tự nó đã đủ làm một section
của paper, và chúng độc lập với việc phần method có thành công hay không.

### D1 — Train 250 epoch là lãng phí; checkpoint được dùng là epoch 20

`best total_loss @ epoch 20/250`. Mọi kết quả đã báo cáo đều đến từ một generator mới train 20 epoch;
230 epoch còn lại bị vứt. → Cắt về ~60 epoch, **rẻ hơn 4×**.

> ⚠️ **Chỉ đúng cho cấu hình 1 adversary tất định.** Run_4 (ensemble + restart) có `best total @ epoch 54`
> và **vẫn đang cải thiện** tại đó — restart giữ cho cuộc chơi không bão hoà sớm. Không áp ngân sách
> 60 epoch cho các biến thể có restart mà chưa kiểm tra lại đường cong.

### D2 — Generator thắng adversary train-time nhưng không khái quát

| epoch | 1 | 10 | **20** | 50 | 100 | 200 | 250 |
|---|---|---|---|---|---|---|---|
| val `ver_loss` | .162 | .055 | **.029** | .047 | .043 | .102 | .100 |
| train `ver_loss` | .294 | .035 | .043 | .009 | .013 | .013 | .032 |

Train `ver_loss` ~0.01 — adversary bị nghiền nát hoàn toàn. Nhưng attacker train lại từ đầu lúc eval
vẫn đạt **0.604**. Không phải adversary yếu — **thông tin vẫn còn nguyên trong ảnh**. Đây là bằng
chứng định lượng cho luận điểm ở mục 0.

### D3 — Ràng buộc utility gần như bất hoạt

| epoch | 1 | 20 | 100 | 250 |
|---|---|---|---|---|
| val `ac_loss` | .1610 | .1539 | .1524 | .1532 |

Suốt 250 epoch `ac_loss` xê dịch 0.008, và *giảm* — generator không chịu áp lực nào từ phía utility.
Trong khi đó AUC của classifier đóng băng tụt 80.5 → 76.2.

**Cơ chế đã đo** (classifier đóng băng `pretrained_classifier.pth`, test set n=25,596, tỷ lệ nhãn dương 7.54%):

| | Ảnh gốc | run_1 biến dạng µ=0.01 | Δ |
|---|---|---|---|
| mean AUC | **0.8050** | 0.7701 | **−0.0349** |
| BCE tổng hợp | 0.20282 | 0.21953 | +0.0167 |
| BCE trên nhãn **dương** | — | — | **+0.4153** |
| BCE trên nhãn **âm** | — | — | **−0.0158** |

Biến dạng phá nhãn dương rất nặng (+0.415) nhưng làm nhãn âm tự tin hơn (−0.016). Vì âm đông gấp
**12.3×**, hai hiệu ứng gần như triệt tiêu: tổn thất thật bị **pha loãng ~25 lần** trước khi tới gradient.
Generator có thể hạ BCE bằng cách đẩy mọi dự đoán về phía âm — đúng lúc AUC (dựa trên thứ hạng) sụp.

Hệ quả: **µ là thứ duy nhất thật sự bảo vệ utility**; loss utility gần như không đóng góp.

Điều này giải thích luôn H1: run_2 refresh critic mỗi epoch → critic thích nghi với biến dạng → tín
hiệu utility còn yếu hơn → utility tụt 0.770 → 0.755. Nhất quán hoàn toàn.

**Hướng sửa (C4):** BCE cân bằng lớp (`pos_weight`) và/hoặc bảo toàn đặc trưng — cả hai đều tránh được
sự pha loãng này.

### D4 — Warp ngẫu nhiên cùng biên độ không mang lại privacy nào

`stochastic_lambda = 1.0` (trường biến dạng thuần ngẫu nhiên, µ=0.01), 10-seed protocol:

| | Re-ID AUC ↓ |
|---|---|
| Ảnh gốc (ours) | 0.8015 ± 0.0270 |
| **Warp ngẫu nhiên λ=1** | **0.8174 ± 0.0287** (n=10, hoàn tất) |
| Biến dạng học được (run_1) | 0.6038 ± 0.0824 |

Warp ngẫu nhiên cho AUC **cao hơn cả ảnh gốc** (chênh lệch nằm trong nhiễu) → **privacy bằng 0**. Toàn bộ
privacy đến từ phần học được. Đây là control mà paper gốc không có, và là bằng chứng trực tiếp cho luận
điểm "đối kháng, không phải lý thuyết thông tin" ở mục 0.

*Neo utility đã hoàn tất: mean AUC ảnh gốc của ta = **0.8050**, khớp chính xác 80.5% của paper.*

---

## 3. Giả thuyết đã đóng (không làm lại)

| # | Giả thuyết | Kết luận |
|---|---|---|
| H1 | ACLoss deepcopy đóng băng → adversary yếu | **Bác bỏ.** Fix làm *tệ hơn* cả hai trục (0.604→0.622; 0.770→0.755). Nay hiểu vì sao: xem D3. |
| H2 | Batch size 16 vs 64 | **Đã xử lý.** Gradient accumulation ×4 (`Agent.py:140`). |
| H3 | Chỉ 1 run vs 10 runs | **Đã xử lý.** `run_snn_multiseed.py`. |
| H4 | Chọn sai checkpoint | **Bác bỏ.** `lowest_ver` cho AUC *cao hơn* (0.641 > 0.604). Dùng `lowest_total`. |
| H5 | Bất đối xứng trong `test_snn` | Không phải deviation, giống paper. |
| T2 | Entropy/confusion loss thay `-log(1-p)` | **Bác bỏ mạnh.** 0.706 — tệ nhất. Lý do giả định ("vanishing gradient") sai: đạo hàm của `-log(1-p)` là `1/(1-p)`, *tăng* khi attacker càng tự tin. |

**Không còn bug fidelity nào để sửa.** Muốn xuống thấp hơn phải đổi **method**.

---

## 4. Đóng góp đề xuất

### ~~C1 — Biến dạng ngẫu nhiên có cấu trúc~~ → **ĐÃ BỊ BÁC BỎ (D4)**
Đã cài đặt đầy đủ (`utils.deform`, tham số `stochastic_lambda`, λ=0 khớp bit-for-bit với baseline). Giữ
lại code vì nó tạo ra dòng ablation D4. **Không dùng làm đóng góp.** Dự đoán: λ trung gian sẽ *có hại*,
vì tổ hợp lồi làm loãng phần nhắm đích `(1-λ)` để đổi lấy entropy vô giá trị.

### C2 (nay là cốt lõi) — Phân bổ ngân sách theo giải phẫu ✅ *đã cài đặt & kiểm chứng*
`F = F_id − M(x) ⊙ F̂` với `M(x)` do U-Net xuất, **ràng buộc `mean(M) = µ`**. Ràng buộc này khiến so
sánh **công bằng theo thiết kế**: tổng ngân sách bằng hệt baseline ở cùng µ, nên mọi cải thiện đến
thuần tuý từ *phân bổ*. Vai trò: giữ utility khi C1 tiêu ngân sách privacy mạnh tay hơn.

### D5 — Adversary mạnh hơn / luôn đổi mới KHÔNG cải thiện privacy

run_4 (K=3 SNN, restart mỗi 25 epoch, warm-up 200 iter) so với run_1:

| | Re-ID AUC ↓ | std | Class. AUC ↑ |
|---|---|---|---|
| run_1 (1 adversary, tất định) | 0.6038 | ±0.0824 | **0.7701** |
| run_4 (C3 ensemble + restart) | 0.6060 | **±0.0478** | 0.7617 |

Privacy **y hệt** (chênh 0.002, sâu trong nhiễu), utility **kém hơn** 0.008 → run_4 bị run_1 thống trị
Pareto. Thu được duy nhất: **giảm 42% phương sai** giữa các seed attacker. Đó vẫn là một đóng góp về
*giao thức đánh giá* (std 0.08 của paper gốc là quá lớn để so sánh tin cậy), nhưng không phải đóng góp
về phương pháp.

### D6 — C2 (budget map) một mình làm utility sụp

| | Re-ID AUC ↓ | Class. AUC ↑ |
|---|---|---|
| run_1 baseline (µ đồng nhất) | 0.6038 ± 0.0824 | **0.7701** |
| **C2 budget map** (µ=0.01, ep58) | *đang eval 10-seed* | **0.7090** |

Utility tụt **6.1 điểm** so với baseline, thấp hơn cả paper (0.762) — dù **tổng ngân sách biến dạng giống
hệt** (`mean(M) = µ` đã kiểm chứng).

**Cách đọc (nhất quán với D3):** budget map cho generator thêm bậc tự do, nhưng critic utility đang bị
pha loãng ~25 lần. Thêm tự do vào một trò chơi có critic hỏng thì generator dùng tự do đó để **khai thác
critic mạnh hơn** — tìm được cách phân bổ vừa thoả mãn BCE vừa phá AUC. Đây không phải bằng chứng chống
lại C2; nó là bằng chứng rằng **C2 cần C4 mới dùng được**.

Dự đoán kiểm chứng được: C2+C4 > C2 một mình trên trục utility. Config `config_anonymization_c2c4.json`.

> Chưa thể kết luận Pareto: cần Re-ID của C2, **và** một điểm baseline µ đồng nhất ở cùng mức utility
> (~0.709, tức µ≈0.02) để so công bằng.

### D7 — Không có phép đo phi-thích-nghi nào dự báo được rủi ro tái định danh

Thử dựng proxy rẻ thay cho giao thức 10-seed (10h): tương đồng cosine trên feature của **ResNet-50
ImageNet đóng băng** — bộ trích đặc trưng độc lập, không nằm trong bất kỳ loss nào nên không thể bị
generator "bắt bài". Script: `proxy_reid.py`.

| | Proxy | Ground truth 10-seed |
|---|---|---|
| Ảnh gốc | 0.7403 | 0.8015 |
| run_1 | 0.6536 | 0.6038 |
| run_2 | 0.6428 | 0.6220 |
| run_3 | 0.6831 | 0.7060 |
| run_4 | 0.6509 | 0.6060 |
| **C2** | **0.6305** (thấp nhất) | **0.7600** (gần cao nhất) |
| λ=1 | 0.7005 | 0.8174 |

**Spearman ρ = 0.464 (p = 0.29)** — không có ý nghĩa. Proxy **đảo ngược** đúng ở ca quan trọng nhất (C2).

**Vì sao:** privacy ở đây được định nghĩa bởi thứ một **bộ học thích nghi** khôi phục được. Feature
tương đồng phi-thích-nghi đo "trông có giống nhau không" — một đại lượng khác hẳn. C2 là bằng chứng
sạch: nó làm ảnh **kém giống hơn** trong không gian ImageNet, trong khi attacker huấn luyện lại khôi
phục identity **tốt hơn bao giờ hết**.

Đây là mở rộng của D4, và nó giải thích vì sao giao thức đắt tiền của lĩnh vực này lại đắt một cách
chính đáng.

**Hệ quả:** mọi proxy dùng được **bắt buộc phải tự huấn luyện một attacker**. Ứng viên tiếp theo: retrain
ngân sách ngắn, 2–3 seed (~20–30 phút thay vì 10h). Phải kiểm định trên **7 điểm ground-truth** trong bảng
trên trước khi tin dùng.

### Kết luận tổng hợp từ D4 + D5: **µ là ràng buộc chặt**

Ba cách làm mạnh trò chơi đối kháng đều thất bại ở cùng một mức ~0.60:

| Can thiệp | Re-ID |
|---|---|
| Warp ngẫu nhiên (D4) | 0.7942 — vô dụng |
| Baseline học được | 0.6038 |
| Adversary mạnh + luôn đổi mới (D5) | 0.6060 |
| Refresh AC critic (H1) | 0.6220 — tệ hơn |
| Entropy loss (T2) | 0.7060 — tệ nhất |

→ **Ở µ cố định, không một cải tiến nào của quá trình huấn luyện đối kháng dịch chuyển được privacy.**
Generator đã vắt gần hết những gì ngân sách cho phép. Chỉ còn hai đòn bẩy thật:

1. **Đổi chỗ tiêu ngân sách** → C2 (phân bổ theo giải phẫu).
2. **Bảo vệ utility tốt hơn để dám tăng µ** → C4 (loss không bị pha loãng).

### ~~C3~~ — Ensemble + restart adversary → **kết quả âm (D5)**, giữ lại vì giảm phương sai
K=3 SNN, restart luân phiên mỗi R epoch + warm-up. Không có nó, generator học đánh lừa một quỹ đạo
tham số thay vì học ngẫu nhiên hoá thật. Sửa mismatch giữa train-time và eval protocol.
*Đã cài đặt; `ver_active_per_step=1` do ràng buộc VRAM 16GB.*

**Cài đặt:** U-Net xuất 3 kênh (2 flow + 1 ngân sách); `M = mu · (1+m) / mean(1+m)` chuẩn hoá theo từng ảnh.
Checkpoint pretrained 2 kênh được nạp một phần, kênh thứ 3 khởi tạo bằng 0 → **bắt đầu đúng từ baseline
µ đồng nhất**. Config: `use_budget_map`. Kiểm chứng: khởi tạo khớp bit-for-bit với baseline (diff `0.0`),
`mean(M)` = đúng µ trên từng ảnh.

### C4 — Loss bảo toàn đặc trưng ✅ *đã cài đặt & kiểm chứng*
Thay BCE bão hoà (D3): khớp feature 1024-d của CheXNet **đóng băng** giữa ảnh thật và ảnh biến dạng.
Tín hiệu dày đặc, gắn với nội dung chẩn đoán. Đây là thành phần nhắm vào **segmentation**: ép bảo toàn
biểu diễn cấu trúc sẽ giữ biên giải phẫu — đúng thứ Dice/IoU đo.

**Cài đặt:** hai knob độc lập trong `ACLoss` — `ac_pos_weight` (chống pha loãng 12.3× đo được ở D3) và
`feature_loss_weight` (MSE trên feature pooled 1024-d). Mặc định `None`/`0.0` khớp bit-for-bit hành vi cũ
(diff `0.0`); feature term = đúng 0 khi so ảnh thật với chính nó.

**Hiệu chỉnh trên dữ liệu thật** (25×2 ảnh validation, µ=0.01, generator run_1):

| Thành phần | Giá trị | So với BCE |
|---|---|---|
| BCE hiện tại | 0.16787 | 1× |
| **feature MSE** | **0.11923** | **0.71×** |
| BCE với `pos_weight=12.3` | 1.59148 | 9.5× |
| `ver_loss` = −log(1−p) | **0.01597** | 0.095× |

→ **`feature_loss_weight = 1.0` đã đúng thang sẵn.** (Ước tính trước đó cần ×50–100 là sai: nó đo trên
ảnh nhiễu, nơi feature vô nghĩa.) Nếu dùng `ac_pos_weight=12.3` thì phải hạ `ac_loss_weight ≈ 0.105` để
giữ thang.

**Phát hiện kèm theo:** ở trạng thái hội tụ `ver_loss` (0.016) nhỏ hơn `ac_loss` (0.168) **10 lần** →
gradient bị `ac_loss` chi phối hoàn toàn, mà D3 đã chứng minh đó chính là tín hiệu bị pha loãng. Generator
đang được dẫn dắt chủ yếu bởi một tín hiệu vô dụng.

---

## 5. Tiêu chí thành công

Mục tiêu **không** phải là "Re-ID < 0.577" — điều đó đạt được tầm thường bằng cách tăng µ.
Tiêu chí là **thống trị Pareto**: cùng Class. AUC thì Re-ID thấp hơn, hoặc cùng Re-ID thì Class. cao hơn.
Bắt buộc phải có **đường cong quét µ** cho cả baseline lẫn phương pháp mới.

Ngoài ra, bổ sung hai phép đo mà paper gốc không có:
- **Top-1 identification rate** trên gallery N bệnh nhân (linkage 1-với-N, sát thực tế hơn AUC theo cặp).
- **Segmentation Dice/IoU/HD95** trên CheXmask, ảnh gốc vs ảnh anonymized.

---

## 6. Lộ trình

| Bước | Việc | Chi phí | Trạng thái |
|---|---|---|---|
| **1.1** | Eval classifier trên ảnh gốc (`none`) — neo mọi số utility, chốt D3 | 20 phút | chờ khe GPU |
| **1.2** | **Control λ=1 (warp ngẫu nhiên thuần)** — xem mục 6.1 | 10-seed eval, **không cần train** | chờ khe GPU |
| **2** | ~~C1 ngẫu nhiên~~ → **đã bác bỏ (D4)**. Chỉ cần chạy nốt seed 4–9 cho bảng | 6h | |
| **2b** | **10-seed eval run_4 (C3 ensemble+restart)** | ~10h | **đang chạy** |
| **3** | **C2 budget map** + **C4 feature loss** — nay là hai đóng góp chính | ~4 run | **bước quyết định** |
| **4** | Quét µ cho baseline + phương pháp mới để dựng đường cong Pareto | ~4 run | |
| **5** | Top-1 identification rate | thấp | |
| **6** | Segmentation downstream trên CheXmask | sau khi có generator tốt | |

Bước 2 là bước quyết định: nếu C1 có tác dụng, ta có paper method. Nếu không, vẫn còn paper chẩn đoán
(D1–D3) + ablation. Biết được sau ~1 ngày, không phải 1 tháng.

### 6.1 Control λ=1 — thí nghiệm rẻ nhất và nhiều thông tin nhất

Ở `stochastic_lambda = 1.0`, trường biến dạng **hoàn toàn ngẫu nhiên** — output của generator bị bỏ qua.
Nghĩa là **không cần train gì cả**: chỉ chạy 10-seed SNN eval (+ eval classifier) trên warp ngẫu nhiên
trơn, cùng µ.

Đây là phép đo chia đôi toàn bộ giả thuyết C1:

- Nếu warp ngẫu nhiên **một mình** đã hạ Re-ID đáng kể dưới 0.604 mà utility không sập → xác nhận luận
  điểm mục 0 (ngẫu nhiên mới là đòn bẩy, không phải phần học được), và đó là một kết quả rất mạnh:
  một control mà paper gốc chưa từng chạy.
- Nếu nó không hạ được → C1 sai, dừng sớm, chuyển sang C2/C4. Tiết kiệm được nhiều ngày train.

Dù kết quả ra sao, đây cũng là một **dòng bắt buộc phải có trong bảng ablation**: nó tách phần đóng góp
của *ngẫu nhiên* khỏi phần đóng góp của *trường biến dạng học được*.

Config: `config_files/config_retrainSNN_randomwarp.json`.

---

## 7. Lệnh

```bash
# 1.1 — classification AUC trên ảnh gốc
python eval_classifier.py --config_path ./config_files/ --config config_eval_classifier_none.json

# Train anonymization
python -u train_architecture.py --config_path ./config_files/ --config <config>.json > <log> 2>&1

# 10-seed SNN eval cho một checkpoint generator
python run_snn_multiseed.py --n_runs 10 \
  --checkpoint ./archive/<run>/generator_lowest_total_loss.pth \
  --out_dir ./archive/retrain_snn_runs_<tag>

# Re-ID baseline trên ảnh gốc, 10 seeds
python run_snn_multiseed.py --n_runs 10 \
  --base_config ./config_files/config_retrainSNN_none.json \
  --out_dir ./archive/retrain_snn_runs_baseline_none

# Mean class. AUC từ file kết quả
python -c "import pandas as pd; print(pd.read_csv('chexnet/results/test/aucs.csv')['auc'].mean())"
```

---

## 8. Hạ tầng & ràng buộc

- GPU: 1× RTX 5070 Ti (16GB). Baseline ~6.5 phút/epoch; ensemble K=3 ~15 phút/epoch (~4 epoch/h).
- **VRAM là ràng buộc cứng.** K=3 SNN cùng trong graph generator → 15.9/16.3GB và thrash.
  `ver_active_per_step=1` giải quyết (10.3GB). Không bật `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`
  (gây `CUDA driver error: device not ready` trên WDDM).
- **Log ghi ra F:**, không ghi C: (C: đầy → Docker daemon kẹt, đã xảy ra một lần).
- Dùng `python -u` để log không bị buffer.
- Ảnh NIH mount tại `/data/images` (112,120 file). CheXmask: `data/chexmask/ChestX-Ray8.csv`
  (RLE, phổi trái/phải + tim, CC-BY, không cần DUA).
- Phương sai giữa seed rất lớn (std 0.03–0.08) → **mọi so sánh phải dùng 10 runs**, không bao giờ kết
  luận từ 1 run.

## 9. Tham chiếu

- Paper gốc: `paper.txt` / `prichexy_paper.pdf` (arXiv:2209.11531, MICCAI 2023).
- CheXmask: Nature Sci. Data 2024, doi 10.1038/s41597-024-03358-1.
