# HANDOVER — PriCheXy-Net Workshop (NEUPS)

Dự án: tái lập và vượt phương pháp anonymization PriCheXy-Net (MICCAI 2023) để PubMed X-quang ngực:
(1) giảm xác suất tái nhận dạng bệnh nhân (Re-ID AUC), (2) giữ/đồng thời giảm prediction utility
(classification), (3) mở rộng đánh giá sang segmentation (utility hình thái).

Tài liệu này là điểm vào để người mới nắm trạng thái, lệnh chạy, và việc còn dở. Về chi tiết cuộc
điều tra, hãy đọc `PLAN.md` ở root trước (đây là bản "diễn biến điều tra" không phải plan tĩnh).

---

## 1. Bàn giao nhanh — câu trả lời 30 giây

Việc đã XONG và kết quả QUAN TRỌNG:

| Mục | Trạng thái |
|---|---|
| Tái lập paper (origin bh5 repro: run_1, `lowest_total_loss`) | Re-ID **0.604 ± 0.082**, class mean AUC **0.770** — nằm trong 1 std của paper (0.577 ± 0.040 / 0.762) |
| Ước lượng nhanh cho mọi comparison | **đã dựng**: frozen ImageNet ResNet-50 proxy, Pearson r=**0.975** trên 5 generator học |
| Bug gradient-accumulation (H2) | **Tìm & sửa** — `zero_grad()` trong loop xoá accumulate → generator bị *under-trained*. Đây là nguồn gốc khoảng cách 0.604 vs 0.577, không phải method. |
| Điều gì *không* cải thiện | H1 (refresh critic per-epoch) 0.604→0.622; entropy/confusion loss (T2) →0.706; ensemble+restart (C3) →0.606 ≈; warp ngẫu nhiên (D4) →0 privacy. |
| Kết luận cốt lõi | Privacy của PriCheX tuân theo cơ chế **đối kháng** chứ không phải lý thuyết thông tin; ở một µ cố định, chỉ cách **dịch chuyển ngân sách biến dạng** (C2) hoặc **giữ utility tốt hơn để dám tăng µ** (C4) mới đổi được cục diện. |

Công việc CÒN DỞ (đã xếp ưu tiên): xem `NEXT_TASKS.md`.

---

## 2. Cấu trúc repo

```
PriCheXy-Net/
├── train_architecture.py          # Train generator anonymization (entry 1)
├── retrain_SNN.py                 # Train attacker SNN trên ảnh đã ẩn danh (entry 2)
├── eval_classifier.py             # Classification mean AUC (downstream utility)
├── run_snn_multiseed.py           # Chạy N seed SNN, gộp mean±std (protocol 10-seed)
├── proxy_reid.py                  # Proxy Re-ID (frozen ImageNet ResNet50), PHÚT-chứa, dùng quét nhanh
├── calibrate_proxy.py             # Kiểm định proxy với real 10-seed (Pearson/Spearman)
├── test_grad_accum.py             # Regression test gradient-accumulation (chặn bug h2)
├── utils/                         # Các module chính: ACLoss, VerificationLoss (ensemble), AdamAcc, deform, train/validate
├── agents/  datasets/  networks/  # Components: generator UNet, SNN, CheXNet, dataset loaders
├── config_files/                  # mọi config thí nghiệm (một config/run)
└── HANDOVER/                      # ← bạn đang tại đây
```

> Không di chuyển các script entry ra sub-directory: chúng phụ thuộc `from utils import`, `from
> agents import` theo đường dẫn tương đối cwd=repo root. Di chuyển sẽ phá lệnh chạy ở PLAN.md.

---

## 3. Quy ước kết quả — nơi nào để đọc số

- Mỗi generation checkpoint: `archive/<run>/generator_lowest_total_loss.pth` + `generator_lowest_ver_loss.pth`.
- Mỗi 10-seed SNN eval: `archive/retrain_snn_runs_<tag>/summary.txt` (`N_runs, AUC_mean, AUC_std, Per_run`).
- Classification: `chexnet/results/<tag>/aucs.csv` (14 pathologies → mean).

**ĐỌC KỸ: mọi con số Re-ID phải là trung bình 10 seeds**, không bao giờ kết luận từ 1 run (std 0.03–0.08).

---

## 4. Các file đáng ngóc

- `PLAN.md` — toàn bộ ý tưởng, bằng chứng D1–D6, hypothesis đóng/bật, lộ trình gốc.
- `archive/.../summary.txt` — mọi con số đã chạy đều lưu ở đây.
- `test_grad_accum.py` — chống tái phát bug gradient accumulation.
- `data/chexmask/ChestX-Ray8.csv` — bộ mask riêng cho segmentation (CC-BY, không cần DUA).

---

## 5. Ranh giới & điều kiện

- GPU 1× RTX 5070 Ti 16GB; VRAM là ràng buộc cứng. Ensemble K=3 `ver_active_per_step` phải =1, không
  bật `expandable_segments` (WDDM crash).
- Baseline ~6.5–12 phút/epoch tuỳ ensemble; 60-epoch run ≈ 4–7h. 10-seed SNN ≈ 1h/run, ~10h.
- **Log phải ghi ra F:**, không ghi C: (C: đầy → Docker daemon kẹt).
- Dùng `python -u` khi bắt đầu run dài để log đừng buffer.

Git: branch `main`, đã push. Xem `.gitignore` — weights `.pth`, `data/`, `archive/` (trừ summary txt),
logs đều không commit.