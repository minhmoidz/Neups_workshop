# NEXT_TASKS — Việc cần làm tiếp cho PriCheXy-Net (xếp theo ưu tiên)

> Đừng kết luận phương pháp nào trước khi hoàn tất T1 (baseline với code đã sửa). Mọi macro kết luận
> trong `PLAN.md` mục 0.5 đang được cảnh báo là chưa vững vì code từng có bug gradient accumulation.
> Số liệu sau "sửa bug" chỉ dùng để xếp ưu tiên, không phải để kết luận.

## Ưu tiên 1 — Bản baseline đúng code (bắt buộc)

### T1. Hoàn tất 10-seed SNN cho baseline_fixed (đang treo ở 4/10)
- Checkpoint: `archive/train_prichexy_net_baseline_fixed/generator_lowest_total_loss.pth`
  (1 adversary, `accumulation_steps=1`, 60 epochs — paper-faithful, đúng code sau khi sửa bug).
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
| Run | Config | Checkpoint | Trạng thái |
|---|---|---|---|
| C2 budget-map | `config_anonymization_c2.json` | `train_prichexy_net_c2_budgetmap` | **eval XONG**: Re-ID 0.760 ± 0.026 (10 seeds), class 0.709 |
| C2+C4 | `config_anonymization_c2c4.json` | — | **chưa train** |
| C4 (feature loss) | `config_anonymization_c4.json` | — | **chưa train** |
| Stoch (λ=0.5) | `config_anonymization_stoch.json` | — | **chưa train** |

- Chạy: `python train_architecture.py --config_path ./config_files/ --config config_anonymization_c2c4.json`
  (60 epochs, ~4–7h mỗi run) rồi `run_snn_multiseed.py` + `eval_classifier`.
- Việc này trả lời: liệu C4 (loss bảo toàn đặc trưng) có cứu được utility mà C2 làm tụt (0.709) không —
  dự đoán `C2+C4 > C2` trên/trục utility (PLAN.md D6).

### T4. Đo Re-ID + class cho control D4 một lần nữa bằng evaluate 10-seed sau khi code ổn
(Warp ngẫu nhiên không cần train; chỉ cần eval lại nếu muốn đóng góp đo trong paper.)

## Ưu tiên 3 (phần đónggóp hình)

### T5. Đường cong µ cho baseline + đề xuất
- Quét µ ∈ {0.005, 0.01, 0.02, 0.04} cho (a) uniform baseline và (b) C2.
- Mục tiêu: kiểm tra "C2 thống trị Pareto" tại bất kỳ µ nào so với baseline.
- Cách đo: train 60ep → proxy (PHÚt) → chỉ applicant tốt mới chạy 10-seed → plot Re-ID vs Class.

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

---

## Nhắc checklist trước khi chạy bất cứ ablation mới
1. `python test_grad_accum.py` → PASS.
2. So config với paper: `accumulation_steps=1`, `ver_ensemble_size=1` (nếu baseline),
    `use_budget_map`, `feature_loss_weight` đúng ý.
3. Ghi log ra ổ D/F (không để C:), dùng `python -u`.
4. Sau khi chạy: nạp `loss_dict.pkl` xem `best total @ epoch ?` rồi mới chọn checkpoint.