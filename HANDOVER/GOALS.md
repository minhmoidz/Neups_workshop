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
| run_1 baseline (code cũ, trước sửa bug) | 0.604 ± 0.082 | 0.770 | mốc "code trước sửa bug" |
| run_2 (H1 refresh) | 0.622 ± 0.052 | 0.755 | refresh critic → thối quality (đã giải thích D3) |
| run_3 (entropy) | 0.706 ± 0.050 | 0.786 | entropy giữ info hơn xóa; privacy tệ nhất |
| run_4 (ensemble+restart) | 0.606 ± 0.048 | 0.762 | phương sai vẫn lớn, chưa hơn baseline (D5) |
| C2 budget map | 0.760 ± 0.026 | 0.709 | utility tụt mạnh khi chưa có C4 |
| control λ=1 ngẫu nhiên | 0.817 ± 0.029 | — | privacy ≈ 0 (D4) |
| ảnh gốc | 0.802 ± 0.027 | 0.805 | upper bound |

---

## Các chỉ ra cụ thể muốn cải thiện hướng tới

1. Làm cho **C2 + C4** thắng baseline ít cột Pareto: dùng C4 feature-retention để tăng Class mà giữ privacy.
2. Giảm phương sai seed (AppD2 ecosystem): công nhận lại mean std của paper (0.03–0.08) đã quá lớn cho so sánh;
   ghi nhận phương pháp giảm phương sai (D5 ensemble+) như 1 cộng lại về protocol đánh giá.
3. Tăng giá trị đóng góp: "repro + ablation" và "hai phép đo mới" đủ làm cơ sở một bài báo tầng
   reproduction/analysis (như đúc kết PLAN.md), độc lập với chuyện phương pháp mới có thắng hay không.
- Nếu C2+C4 cho kết quả rõ ràng; deliverable cuối: (a) reproduction faithful + ablation table;
  (b) nếu được: phương pháp vượt baseline trên Pareto; (c) số segmentation downstream (Dice/IoU) kèm.