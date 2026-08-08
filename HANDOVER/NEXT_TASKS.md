# NEXT_TASKS — Việc cần làm tiếp cho PriCheXy-Net (xếp theo ưu tiên)

> **Mọi so sánh "cải thiện" phải neo trên baseline đúng code (T1/T2).** Tất cả số Re-ID/Class ghi trong
> PLAN.md mục 0.5 đều sinh ra từ code *có bug* `zero_grad()` trong gradient accumulation — kể cả run_1
> (0.604), run_2, run_3, run_4, C2 (0.760/0.709). Chúng chỉ dùng để *định hướng*, **không** được dùng làm
> baseline so sánh để kết luận "phương pháp này cải thiện hay không".

---

## Nguyên tắc cải thiện (đọc trước khi làm gì)

1. **Biến số tuân theo hướng**: Re-ID AUC ↓ = privacy tốt hơn; Class AUC ↑ = utility tốt hơn.
   Một phương án "cải thiện" CHỈ khi nó thắng baseline trên ≥1 trục và không thua trục kia (Pareto),
   đo trên **cùng 10-seed protocol + cùng tập test**, so với **cùng µ**.
2. **Baseline chuẩn = `baseline_fixed`** (1 adversary, `accumulation_steps=1`, 60 epochs, code sau sửa bug),
   cặp (Re-ID, Class) lấy từ **T1 + T2**. Trước khi T1/T2 xong, không kết luận gì về "so với paper 0.577 / 0.762".
3. **Không tin số cũ vội**: nếu một khẳng định trong PLAN.md dựa trên run_1/run_2/.../C2 (kết quả cũ
   hoặc chưa chạy 10-seed đúng), phải chạy lại/retrain mới (T8/T9) trước khi dùng làm bằng chứng.
4. **µ-sweep làm nổi bật Pareto**, không phải để "đạt mốc số to". Chỉ tăng µ nếu thêm được giá trị
   (ví dụ giữ class cao mà Re-ID giảm) — nếu chỉ đổi chỗ đánh đổi thì không gọi là cải thiện.
5. **Rẻ trước, đắt sau**: mọi ý tưởng cải thiện phải lọc qua `proxy_reid.py` (r=0.975 với 10-seed) trước khi
   cam kết 10-seed (≈10h) + retrain (≈4–7h).

---

## Bước 0 — Chuẩn bị khi mới clone/pull (làm 1 lần)

1. **Data ảnh**: xem `HANDOVER/DATA.md` — 112,120 PNG đặt ở địa điểm mà `image_path` trong config
   trỏ tới (mặc định `/data/images/`). Số file phải = 112,120.
2. **Dependencies**: `pip install -r requirements.txt` (torch phù hợp CUDA server).
3. **Sửa `image_path`**: các config đang để `"/data/images/"`; nếu server dùng path khác, sửa đồng bộ
   4 loại config: `config_pretrain`, `config_anonymization_*`, `config_retrainSNN`, `config_eval_classifier*`.
4. **Checkpoints có sẵn trong git (không cần làm gì)**:
   - `networks/*.pth` (pretrained classifier/generator/verifier — LFS, pull tự lấy).
   - `archive/train_prichexy_net_baseline_fixed/generator_lowest_total_loss.pth` (LFS).
   - Mọi kết quả số `archive/retrain_snn_runs_*/summary.txt` (đã commit).
5. **Checkpoint KHÔNG có trong git** (phải tự retrain nếu cần, xem T9): generator của
   `c2_budgetmap`, `run_1`, `run_2_h1fix`, `run_3_entropy`, `run_4_ensemble`.
6. Test nhanh data/setup: chạy smoke 2 epochs bằng `python train_architecture.py --config_path ./config_files/ --config config_smoke_acc64.json`.

---

## Ưu tiên 1 — Bản baseline đúng code (bắt buộc)

### T1. Hoàn tất 10-seed SNN cho baseline_fixed (đang treo ở 4/10)
- Checkpoint: `archive/train_prichexy_net_baseline_fixed/generator_lowest_total_loss.pth`
  (1 adversary, `accumulation_steps=1`, 60 epochs — paper-faithful, đúng code sau khi sửa bug).
- **Checkpoint này CÓ trong git (LFS)** khi bạn clone/pull repo — không cần retrain.
  Các generator khác (`c2`, `run_1`, `run_2`, `run_3`, `run_4`) **không** trong repo, phải tự retrain
  theo T9 nếu cần dùng.
- Đã có: SNN eval seed 0–3 (4/10). Chạy tiếp 6 seed còn lại:

```bash
cd /workspace
python run_snn_multiseed.py --n_runs 6 \
  --checkpoint ./archive/train_prichexy_net_baseline_fixed/generator_lowest_total_loss.pth \
  --out_dir ./archive/retrain_snn_runs_baseline_fixed --start_seed 4
```

- Verify: `summary.txt` có `N_runs: 10`.
- Ý nghĩa: con số Re-ID **thật sự so được với paper 0.577** (trước kia 0.604 là của mã có bug).
  Nếu hội tụ về ~0.58–0.60 → xác nhận paper reproducible; nếu thấp hơn hẳn (≈0.55) → bug đúng
  thủ phạm, và cách tái hiện trong paper cần nói rõ batch protocol.

### T2. Classification eval cho baseline_fixed
```bash
python eval_classifier.py --config_path ./config_files/ --config config_eval_classifier_baseline_fixed.json
```
- Verify: `chexnet/results/test_baseline_fixed/aucs.csv` → mean AUC.
- Cùng với T1 tạo ra cặp (privacy, utility) cho baseline sau-sửa — mới duy nhất có ý nghĩa tham chiếu.

## Ưu tiên 2 (đánh giá các phụ-release ablation)

### T3. Hoàn tất training + eval cho các ablation còn thiếu
> ⚠ Số liệu trong bảng này (trừ cột "Trạng thái") là của **code cũ / chưa chạy 10-seed đúng** — chỉ để
> tham khảo hướng, **không phải bằng chứng cải thiện**. Muốn dùng phải retrain lại trên code đã sửa (T9).

| Run | Config | Checkpoint đúng-code | Trạng thái |
|---|---|---|---|
| C2 budget-map | `config_anonymization_c2.json` | cần retrain | eval cũ (code bug): Re-ID 0.760, class 0.709 — **không dùng làm mốc** |
| C2+C4 | `config_anonymization_c2c4.json` | — | **chưa train** |
| C4 (feature loss) | `config_anonymization_c4.json` | — | **chưa train** |
| Stoch (λ=0.5) | `config_anonymization_stoch.json` | — | **chưa train** |

- Chạy: `python train_architecture.py --config_path ./config_files/ --config config_anonymization_c2c4.json`
  (60 epochs, ~4–7h mỗi run) rồi `run_snn_multiseed.py` + `eval_classifier`.
- Việc này trả lời: liệu C4 (loss bảo toàn đặc trưng) có cứu được utility hay không — **so với baseline
  ở T1/T2, không phải so với số C2 cũ**. Kết luận chỉ hợp lệ sau khi có baseline đúng code (T1+T2).

### T4. Đo Re-ID + class cho control D4 một lần nữa bằng evaluate 10-seed sau khi code ổn
(Warp ngẫu nhiên không cần train; chỉ cần eval lại nếu muốn đóng góp đo trong paper.)

## Ưu tiên 3 (phần đónggóp hình)

### T5. Đường cong µ cho baseline + đề xuất
- Quét µ ∈ {0.005, 0.01, 0.02, 0.04} cho (a) baseline_fixed (đúng code) và (b) phương án đang thử (sau T3).
- Mục tiêu: kiểm tra phương án thử có thống trị Pareto **tại bất kỳ µ nào** so với baseline — trên
  cùng µ, chứ không phải trộn µ khác nhau rồi so.
- Cách đo: train 60 ep (đúng code) → proxy (phút) → chỉ ứng viên tốt mới chạy 10-seed → plot Re-ID vs Class.

### T6. Top-1 identification rate (khi có generator tạm ổn định)
- Linkage 1-với-N trên gallery N bệnh nhân; phép đo này paper không có — thêm giá trị.

## Ưu tiên 4 (downstream segmentation — dự kiến giá trị lớn)

### T7. Segmentation Dice/IoU/HD95 trên CheXmask
- Dataset: `data/chexmask/ChestX-Ray8.csv` (RLE phổi trái/phải + tim, CC-BY, có sẵn local).
- Trạng thái: thư mục `segmentation/` mới chỉ có `_test_net.py` — **chưa có pipeline đo**.
- Cần: (1) parse mask RLE → array; (2) chạy segmenter (CheXNet/U-Net đơn giản) trên ảnh gốc và ảnh
  anonymized (cùng đo lường); (3) so đoán Dice/IoU/HD95.
- Có thể song song với T5/T6 (chỉ cần generator ổn định; bộ mask đã có sẵn).

## Kéo dài: tooling & rigorous
- Giữ bộ regression test (T8): chạy `python test_grad_accum.py` trước mỗi baseline/ablation mới;
  cân nhắc test tương tự cho `ACLoss.refresh()` và `VerificationLoss` ensemble.

### T9. Retrain generator khi cần checkpoint chưa có trong git
Dùng để tái tạo `c2_budgetmap` / `run_1` / `run_3` / ... khi agent cần chạy eval cho các run đó:

```bash
python -u train_architecture.py --config_path ./config_files/ --config config_anonymization_run1.json \
  > logs/retrain_run1.log 2>&1
# sau khi xong, tìm checkpoint tốt nhất:
#   archive/<experiment_description>/generator_lowest_total_loss.pth
# rồi chạy 10-seed SNN (giống T1) + eval_classifier (giống T2)
```
- Run thường mất 4–7h (60 epochs, GPU RTX). Nếu server không đủ GPU, chỉ dùng checkpoint có sẵn trong git.

---

## Nhắc checklist trước khi chạy bất cứ ablation mới
1. `python test_grad_accum.py` → PASS.
2. So config với paper: `accumulation_steps=1`, `ver_ensemble_size=1` (nếu baseline),
    `use_budget_map`, `feature_loss_weight` đúng ý.
3. Ghi log ra ổ D/F (không để C:), dùng `python -u`.
4. Sau khi chạy: nạp `loss_dict.pkl` xem `best total @ epoch ?` rồi mới chọn checkpoint.