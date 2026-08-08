# DATA — Dữ liệu dự án PriCheXy-Net

Tài liệu này mô tả: (1) các khối dữ liệu dự án cần, (2) cách chia train/val/test trong code,
(3) nời tải dữ liệu gốc, (4) checklist khi chạy trên server (không dùng Docker).

---

## 1. Tổng quan 5 khối dữ liệu

| # | Khối | Dùng cho | Loader đọc từ đâu |
|---|---|---|---|
| 1 | Ảnh NIH ChestX-ray14, 112,120 ảnh PNG | Anonymization train/val/test, SNN re-id, classifier | `image_path` trong config (mặc định `/data/images/`) |
| 2 | `Data_Entry_2017_v2020.csv` (meta ảnh, 112,120 dòng) | Nhãn 14 bệnh cho auxiliary classifier | `./Data_Entry_2017_v2020.csv` (root repo) |
| 3 | Các file liệt kê trong `image_pairs/` | Split số liệu + danh sách cặp ảnh cho SNN | `./image_pairs/...` |
| 4 | `data/chexmask/ChestX-Ray8.csv` (~2GB) | Segmentation downstream (chưa làm, xem T7) | `./data/chexmask/` |
| 5 | Pretrained checkpoints | Khởi tạo classifier/SNN | `networks/*.pth` (lfs) |

---

## 2. Ảnh NIH ChestX-ray14

- **112,120 file `.png`**, tên dạng `00001661_000.png` (8 chữ số patient + 3 số follow-up).
- **~30,805 bệnh nhân** (`Patient ID` trong Data_Entry).
- Ảnh gốc Resize → **256×256** khi train (1 kênh, convert gray).

### 2.1 Nơi tải

NIH ChestX-ray14 là bộ dữ liệu công cộng, có 2 nguồn phổ biến:

| Nguồn | Cách tải |
|---|---|
| **Kaggle** (khuyến nghị nhất) | `kaggle datasets download -d nih-chest-xrays/nih-chest-xrays-dataset`, hoặc vào trang Kaggle tìm "NIH Chest X-Ray" bản 122k images |
| **NIH official GitHub release** | 12 file zip `images_001.zip`–`images_012.zip` được NIH đăng; tải toàn bộ rồi giải nén vào chung 1 thư mục |

> NIH không còn cho phép tải trực tiếp qua trỏ link tĩnh (cần phải đăng ký box.com); do đó
> **Kaggle là đường chắc ăn nhất** vì ảnh giữ nguyên.

Cách chuẩn bị sau khi tải (dù nguồn nào):

```bash
mkdir -p images && cd images
# (giải nén 12 shard vào đây)
ls *.png | wc -l   # PHẢI = 112120
```

---

## 3. Chia dữ liệu trong code

### 3.1 Split pretrain generator (`datasets/DatasetPretrain.py`)
- `image_pairs/train_val_list.txt`: 86,524 ảnh
  - `[:75708]` → **train** (75,708)
  - `[75708:]` → **validation** (10,816)
- `image_pairs/test_list.txt`: 25,596 ảnh → **testing**

→ Tổng 86,524 + 25,596 = **112,120** khớp với số ảnh.

### 3.2 Split cặp ảnh (anonymization + Re-ID, `datasets/Dataset.py`)
| File | Số cặp | Phase |
|---|---|---|
| `image_pairs_training_10000.txt` | 10,000 | training |
| `image_pairs_validation_2000.txt` | 2,000 | validation |
| `image_pairs_testing_5000.txt` | 5,000 | testing (Re-ID eval) |

**Mỗi dòng** (3 cột, whitespace):
```
00001661_000.png  00001661_001.png  1.0
00016972_001.png  00016972_013.png  1.0
```
- Cột 3 là `1` nếu **cùng patient**, `0` nếu **khác patient** (cho Re-ID SNN học + đánh giá).
- 10,000 cặp train gồm ~5,000 cùng + 5,000 khác (kiểm tra thực tế: 5000/5000).

### 3.3 Nhãn aux-classifier (14 class)
- Đọc từ `Data_Entry_2017_v2020.csv`, cột `Finding Labels`; bệnh nối nhau bằng `|`.
- `Dataset.PRED_LABEL` 14 class theo thứ tự code: Atelectasis, Cardiomegaly, Effusion, Infiltration,
  Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis,
  Pleural_Thickening, Hernia.
- Khớp với `chexnet` 14 pathologies (classifier eval).

---

## 4. CheXmask (segmentation downstream)

- `data/chexmask/ChestX-Ray8.csv` (~2GB): cột `Image Index`, RLE mask `Left Lung`, `Right Lung`,
  `Heart`, kèm `Height`/`Width` (1024×1024).
- Nguồn: **CheXmask** của nhóm (HuggingFace / Google Drive / GitHub theo repo chính thức).
- Chưa dùng trong code train hiện tại; chỉ cần khi mở T7 segmentation.

---

## 5. Data_Entry_2017_v2020.csv — chú ý

- 112,120 dòng, mỗi ảnh 1 dòng, cột: `Image Index`, `Finding Labels`, `Follow-up #`, `Patient ID`,
  `Patient Age`, `Patient Gender`, `View Position`, ...
- Cột `Patient ID` có value số, không phải chuỗi → phân biệt 30,805 id duy nhất.
- Bản tải NIH gốc `Data_Entry_2017.csv` chỉ CÓ 8 cột (không có `Patient ID`). Bản `v2020` trong
  repo này đã thêm `Patient ID`/`Patient Age`/... — nếu tải từ Kaggle, chọn bản có `Patient ID`.

---

## 6. Server setup (không Docker)

### B1. Cài ảnh + metadata
```bash
# giả định ảnh tại /data/images (đúng image_path)
ls /data/images/*.png | wc -l   # → 112120
cp image_pairs/...               # các file split đã có trong repo (git)
cp Data_Entry_2017_v2020.csv ./ # tại root repo (loader đọc relative ./)
```

### B2. Cấu hình `image_path`
Mặc định mọi config = `"/data/images/"`. Thay cho đúng máy của bạn (VD server dùng `/data/nih/`):

```bash
# trong 4 loại config:
#   config_pretrain.json
#   config_anonymization_*.json
#   config_retrainSNN.json
#   config_eval_classifier*.json
sed -i 's#"/data/images/"#"/your/path/"#g' config_files/*.json
```

### B3. Cài dependencies
```bash
pip install -r requirements.txt   # torch, torchvision, pandas, numpy, Pillow, scipy ...
```
(Phụ thuộc CUDA version của server; dùng bản torch phù hợp.)

### B4. Sanity test vài epoch
```bash
python train_architecture.py --config_path ./config_files/ --config config_smoke_acc64.json
# hoặc smoke 1–2 epoch để kiểm tra data load
```
Nếu chạy vài epoch không lỗi data → data chuẩn.

---

## 7. Quy ước file không được thiếu

| File | bắt buộc? |
|---|---|
| 112,120 tấm PNG | **bắt buộc** cho mọi thí nghiệm |
| `Data_Entry_2017_v2020.csv` | **bắt buộc** (mọi phase, kể cả eval_classifier đọc labels) |
| `image_pairs/*.txt` (5 files) | **bắt buộc**, đã track trong repo |
| `data/chexmask/ChestX-Ray8.csv` | chỉ khi chạy T7 segmentation |
| `networks/pretrained_classifier.pth` | chỉ khi eval classifier (có sẵn LFS hoặc tải CheXNet) |

---

## 8. Ghi chú triển khai trên server

- Dữ liệu ở môi trường này (`F:\datasets\...\images`) là bản đầy đủ 112,120 ảnh, đã mount `/data/images`
  trong Docker. Khi cài server không Docker, tự lấy ảnh theo mục 2.
- Nếu server không có mạng ngoài (chỉ proxy nội bộ): tải Kaggle/NIH trên máy có mạng rồi upload bằng
  `scp -r ./images user@server:/data/` (hoặc rsync).