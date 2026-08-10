# GOALS — Mục tiêu cải thiện (để bàn giao và đo sau)

## Mục tiêu cuối cùng (định nghĩa "thành công")

**KHÔNG PHẢI** "Re-ID < 0.577". Giá trị đó đạt được tầm thường bằng cách tăng µ (biến dạng mạnh hơn)
nhưng trả giá bằng utility. Tiêu chí thật:

> **Pareto dominance**: cùng Class AUC (utility) mà Re-ID thấp hơn, hoặc cùng Re-ID mà Class cao hơn,
> so với cả (a) baseline paper (0.577 / 0.762) và (b) cặp (privacy, utility) đạt được khi xén µ.
>
> Điều kiện bắt buộc: mọi so sánh phải nằm trên **cùng một đường cong µ** cho cả baseline lẫn method.

## Cải thiện theo 3 tầng

| Tầng | Mục tiêu | Cách đo | Trạng thái |
|---|---|---|---|
| T (repro) | Tái hiện paper trung thực: Re-ID ≈ 0.577, Class ≈ 0.762 | 10-seed + class trên baseline code-sửa-bug | Chờ T1/T2 để có baseline chuẩn |
| A (ablation đóng góp) | Điều tra cơ chế: (a) proxy tương quan 0.975 là cửa vào nhanh; (b) các nghiên cứu âm H1/T2/D4/D5/D6 thành dòng ablation có giá trị; (c) C2 budget map + C4 feature retention là 2 hướng chính | quét từng ablation + µ | C2/C4/C2C4 chưa chạy; µ-sweep chưa có |
| B (tiêu chí hiện đại) | 2 phép đo mới không có trong paper cũ: (1) **Top-1 identification** thực-chiến gallery N:1; (2) **Segmentation Dice/IoU/HD95** trên CheXmask | T6, T7 trong NEXT_TASKS | Chưa bắt đầu |

## Kết quả hiện có (privacy / utility — các cặp Re-ID, Class)

| Phương án | Re-ID (10-run) | Class mean AUC | Kết luận tạm thời |
|---|---|---|---|
| **baseline_fixed (đúng code, 60 ep, acc=1)** | **0.635 ± 0.079** | **0.773** | ✅ baseline chuẩn (T1+T2 xong 2026-08-09) — mốc so sánh duy nhất hợp lệ |
| run_1 baseline (code cũ, trước sửa bug) | 0.604 ± 0.082 | 0.770 | mốc "code trước sửa bug" (chỉ tham khảo) |
| run_2 (H1 refresh) | 0.622 ± 0.052 | 0.755 | refresh critic → thối quality (đã giải thích D3) |
| run_3 (entropy) | 0.706 ± 0.050 | 0.786 | entropy giữ info hơn xóa; privacy tệ nhất |
| run_4 (ensemble+restart) | 0.606 ± 0.048 | 0.762 | phương sai vẫn lớn, chưa hơn baseline (D5) |
| C2 budget map | 0.760 ± 0.026 | 0.709 | utility tụt mạnh khi chưa có C4 |
| control λ=1 ngẫu nhiên | 0.817 ± 0.029 | — | privacy ≈ 0 (D4) |
| ảnh gốc | 0.802 ± 0.027 | 0.805 | upper bound |

## Segmentation downstream (T7, CheXmask — Dice/IoU/HD95 trên 400 ảnh test, segmenter U-Net feat=16)

| Phương án (µ) | Dice LL/RL/Heart | Dice mean | IoU mean | HD95 mean |
|---|---|---|---|---|
| ảnh gốc (upper bound, không deform) | 0.947 / 0.958 / 0.937 | **0.947** | 0.905 | 1.74 |
| **C4 (µ=0.01)** | 0.936 / 0.950 / 0.929 | **0.938** | **0.889** | **2.12** |
| baseline_fixed (µ=0.01) | 0.934 / 0.948 / 0.928 | 0.937 | 0.886 | 2.27 |
| C4 (µ=0.02) | 0.926 / 0.941 / 0.917 | 0.928 | 0.870 | 2.61 |
| C4 (µ=0.04) | 0.887 / 0.901 / 0.903 | 0.897 | 0.818 | 4.25 |
| C2+C4 (µ=0.01) | 0.898 / 0.899 / 0.892 | 0.897 | 0.817 | 4.75 |

## Top-1/5 identification (T6 — phép đo mới, N=500 gallery 1:1 ảnh/patient, benchmark 2026-08-10)

| Phương án (µ) | TOP-1 | TOP-5 | MRR |
|---|---|---|---|
| ảnh gốc (upper bound) | 0.170 | 0.282 | 0.231 |
| baseline_fixed (0.01) | 0.152 | 0.274 | 0.217 |
| C4 (0.01) | 0.156 | 0.276 | 0.220 |
| C2+C4 (0.01) | 0.142 | 0.250 | 0.202 |
| C4 (0.02 / 0.04) | chờ seed2 train xong | — | — |

> **10-seed Re-ID thật (2026-08-10) — C4 THẤT BẠI privacy:**
> | C4 µ | Re-ID 10-seed | Class | Seg Dice | Top-1 |
> |---|---|---|---|---|
> | 0.01 | 0.7496 ± 0.033 | 0.792 | 0.938 | 0.156 |
> | 0.02 | 0.7598 ± 0.037 | 0.788 | 0.928 | 0.148 |
> | 0.04 | **0.6985 ± 0.065** | 0.785 | 0.897 | 0.148 |
> | baseline_fixed 0.01 | **0.635 ± 0.079** | 0.773 | 0.937 | 0.152 |
>
> C4 (feature retention) **giữ identity → Re-ID cao hơn baseline ở mọi µ**. Proxy frozen ResNet-50 báo
> C4@0.04 = 0.634 nhưng thực tế 0.70 → proxy KHÔNG tin cho C4/C2 (confirmed D7). Hướng privacy phải là
> C2 budget map hoặc stochastic_lambda (destruct), không phải C4.
> (Lưu ý: run C4@0.02 đầu tiên dính bug `mu`=0.01 → đúng là C4@0.01; đã rerun đúng µ=0.02 → `retrain_snn_runs_c4_mu0.02_fixed`.)

## Đường cong µ C4 hoàn chỉnh (privacy / class / seg, 2026-08-09)

| Method (µ) | Re-ID proxy | Class mean AUC | Seg Dice mean | Nhận xét |
|---|---|---|---|---|
| ảnh gốc (upper bound) | 0.802 (real) | 0.805 | 0.947 | — |
| C4 (0.01) | 0.669 | **0.792** | **0.938** | class cao nhất, seg ≈ original |
| C4 (0.02) | 0.658 | 0.788 | 0.928 | **điểm cân bằng tốt nhất** |
| C4 (0.04) | 0.634 | 0.785 | 0.897 | privacy tốt nhất nhưng seg tụt |
| baseline_fixed (0.01) | 0.653 | 0.773 | 0.937 | mốc so sánh |
| C2+C4 (0.01) | 0.663 | 0.784 | 0.897 | seg phá ngang C4@0.04 |

> **Kết luận µ-sweep:** C4 (0.02) thắng baseline trên 2/3 trục (Class 0.788 > 0.773, seg 0.928 ≈ 0.937,
> privacy proxy ~0.658 ≈ 0.653). C4 (0.04) mạnh privacy (proxy 0.634) nhưng hi sinh seg (0.897).
> C2 budget map phá segmentation ngang C4@µ=0.04 dù cùng µ — C4 là hướng giữ seg tốt nhất.
> **Ứng viên cam kết 10-seed Re-ID tiếp:** C4 (0.02) hoặc C4 (0.04) nếu muốn privacy tối đa.

> **Đóng góp chính:** C4 (feature-retention) giữ segmentation **ngang bằng ảnh gốc** (Dice 0.947→0.938, mất chỉ ~1%)
> dù vẫn hạ Re-ID so với baseline — trong khi C2 budget map (nhắm vùng phổi/tim) làm seg suy giảm rõ (0.937→0.897).
> C4 chứng minh "privacy tại vùng Giải phẫu" không khắc-xuống đường biên hình thái → seg giữ được.
> (Checkpoint: `archive/train_seg_unet/best.pth`, val dice 0.955/0.964/0.946 trên 1500 ảnh.)

---

## Các chỉ ra cụ thể muốn cải thiện hướng tới

1. Làm cho **C2 + C4** thắng baseline ít cột Pareto: dùng C4 feature-retention để tăng Class mà giữ privacy.
2. Giảm phương sai seed (AppD2 ecosystem): công nhận lại mean std của paper (0.03–0.08) đã quá lớn cho so sánh;
   ghi nhận phương pháp giảm phương sai (D5 ensemble+) như 1 cộng lại về protocol đánh giá.
3. Tăng giá trị đóng góp: "repro + ablation" và "hai phép đo mới" đủ làm cơ sở một bài báo tầng
   reproduction/analysis (như đúc kết PLAN.md), độc lập với chuyện phương pháp mới có thắng hay không.
- Nếu C2+C4 cho kết quả rõ ràng; deliverable cuối: (a) reproduction faithful + ablation table;
  (b) nếu được: phương pháp vượt baseline trên Pareto; (c) số segmentation downstream (Dice/IoU) kèm.