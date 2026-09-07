# project-dl — NABirds

Phân loại chim chi tiết (fine-grained) trên **NABirds v0** (Cornell Lab of Ornithology /
Visipedia): 48,562 ảnh, **555 lớp lá** → **404 loài** → **22 bộ**. Toàn bộ chạy trên một
máy cá nhân: **RTX 5070 12GB, Windows 11**.

Repo gồm hai dự án đã hoàn thành và một dự án chưa bắt đầu:

| # | Dự án | Trạng thái | Kết quả chính |
|---|---|---|---|
| 1 | Binary classification (CNN/ResNet) | ⛔ **bỏ** | Quyết định D13 — không cho kết luận nào mà bài 555 lớp chưa có |
| 2 | Multi-class 555 lớp, có giám sát | **xong** | **44 run qua 8 pass**. ViT-B/16 IN21k @**448** — **top-1 90.03%** (cách TransFG 0.77 điểm mà không dùng module FGVC nào) |
| 3 | Zero-shot kiểu CLIP / METS (trục biến thể: **compositional** ZS) | **xong** | xem [§ Dự án 3](#dự-án-3--zero-shot-kiểu-clip--mets) |

> **Đọc gì trước:** nếu bạn (hoặc một AI agent) mới vào repo, đọc hết file này là đủ nắm
> bối cảnh, kết quả và các bẫy đã gặp. Chi tiết đầy đủ nằm trong [`docs/`](#tài-liệu-chi-tiết).

### Trạng thái vận hành hiện tại — 2026-09-07

Máy đang chạy thêm một ablation augmentation cho BioCLIP:

```text
PID 26092: python src/train.py --model bioclip --run-name bioclip_224_cmo
            --epochs 25 --batch-size 64 --patience 8 --val-frac 0.15
            --workers 12 --erasing-p 0 --cmo-p 0.50
```

Snapshot kiểm tra mới nhất: epoch 18/25 đã hoàn tất với `val-top1 = 89.80%`;
epoch 19 đang train và checkpoint `runs/bioclip_224_cmo/best.pt` còn được cập nhật. Chưa có
`results/bioclip_224_cmo/summary.json`, nên **chưa được ghi là kết quả test cuối**;
chờ run kết thúc rồi mới chạy bước report và cập nhật bảng chính.

---

## Quickstart

```bash
pip install timm open_clip_torch peft pyarrow

# --- chuẩn bị dữ liệu (chạy 1 lần) ---
python src/prepare_images.py                       # pre-resize -> nabirds/images_r448
NAB_SHORT_SIDE=512 python src/prepare_images.py    # cache 512, bắt buộc cho mọi run @448
```

### Tám pass đã chạy — chạy lại theo đúng thứ tự này

```bash
bash src/run_all.sh      # pass 1: sweep @224/299 - cnn_scratch, resnet, inception, ViT
bash src/run_448.sh      # pass 2: độ phân giải 448 (cần cache images_r512)
bash src/run_cnn2.sh     # pass 3: CNN hiện đại - ConvNeXt, EfficientNetV2 + cầu nối AdamW
bash src/run_more.sh     # pass 4: BioCLIP, 5 CNN cơ bản, ViT@448, ensemble+TTA
bash src/run_more2.sh    # pass 5: trục thời gian 2012-2022 + ViT-S
bash src/run_aug.sh      # pass 6a: ablation augmentation (erasing/rotate/hue/100ep/cutmix)
bash src/run_aug2.sh     # pass 6b: CMO + bộ ba cho model mạnh
bash src/run_partq.sh    # pass 8: part-query decoder (encoder-decoder + cross-attention)
bash src/run_cub.sh      # pass 7: chuyển giao NABirds -> CUB-200-2011
```

Mọi script sweep đều **resume được**: run nào đã có `results/<tên>/summary.json`
thì bị bỏ qua; đặt `FORCE=1` để chạy lại. Khoá chống chạy chồng ở
`src/sweep_lock.sh` (mỗi sweep một PID file riêng trong `runs/`).

> **KHÔNG sửa file trong `src/*.py` khi một sweep đang chạy** — bẫy #18. Trên
> Windows, DataLoader worker được spawn: nó import lại module (code MỚI) rồi
> unpickle object dataset (dựng bằng code CŨ) ⇒ `AttributeError` giết run ở giữa.

### Chạy một model đơn lẻ

```bash
python src/train.py --model vit_b_16_in21k --epochs 25 --batch-size 64
```

Cờ đáng nhớ của `src/train.py` (mặc định giữ nguyên hành vi gốc):

| cờ | mặc định | dùng khi nào |
|---|---|---|
| `--img-size 448 --image-dir images_r512` | 224 | đổi độ phân giải. **Bắt buộc đi cùng nhau** |
| `--grad-checkpointing` | tắt | ViT @448 trên 12GB (784 token/ảnh). Đổi ~30% thời gian lấy VRAM |
| `--optimizer adamw --lr-backbone 1e-4 --lr-head 1e-3 --weight-decay 0.05` | SGD | **bắt buộc** cho ConvNeXt / EfficientNetV2 / Swin — xem **D14** |
| `--lora-rank 16` | 0 | chỉ hỗ trợ `vit_b_16_in21k` |
| `--erasing-p 0` | 0.25 | tắt RandomErasing (E1) |
| `--rotate 15` | 0 | xoay ±N độ (E2) |
| `--hue 0.10` | 0.02 | **cẩn thận — va vào D3**, chỉ dùng làm đối chứng (E4) |
| `--cutmix-alpha 1.0 --mixup-alpha 0.2 --mix-prob 0.5` | 0 | trộn ảnh. **Chỉ có nghĩa khi ≥100 epoch** (E3) |
| `--cmo-p 0.5 --cmo-tail-max 30` | 0 | CMO: dán chim lớp hiếm lên nền lớp nhiều ảnh |

```bash
# part-query decoder (encoder-decoder + attention)
#   encoder dùng NGUYÊN trọng số có sẵn; chỉ decoder 15.14M là train mới
python src/partquery.py --model vit_b_16_in21k \
    --encoder-ckpt runs/vit_b_16_in21k_224/best.pt --freeze-encoder \
    --parts 12 --dec-layers 2 --epochs 15

# chuyển giao sang CUB (--init imagenet | nabirds)
python src/cub_transfer.py --model vit_b_16_in21k --init nabirds \
    --ckpt runs/vit_b_16_in21k_224/best.pt
```

### Phân tích — chạy sau khi có checkpoint, KHÔNG train gì

```bash
python src/report.py --save               # bảng mọi run  -> results/comparison.csv|.md
python src/aug_report.py                  # tách lớp đuôi/khó -> results/aug_comparison.csv|.md
python src/ensemble.py --cache --report   # TTA + ensemble    -> results/ensemble/
python src/logit_adjust.py                # logit adjustment  -> results/logit_adjust.csv|.md
python src/cub_transfer.py --report       # bảng CUB          -> results/cub/comparison.csv
python src/report.py --run <tên_run>      # chi tiết 1 run
```

> **Khi đánh giá augmentation phải đọc `aug_report.py`, không phải `report.py`.**
> 113 lớp đuôi chỉ chiếm ~10% ảnh test nên top-1 tổng che mất hiệu ứng thật (**D17**).

### Dự án 3 — zero-shot

```bash
python zeroshot/src/build_variants.py            # 555 lá  -> data/leaf_variants.csv
python zeroshot/src/build_taxonomy.py            # 404 loài -> data/species_taxonomy.csv (GBIF)
python zeroshot/src/contamination.py             # đối chiếu TreeOfLife-10M
python zeroshot/src/build_tol_taxonomy.py        # chuỗi taxonomy BioCLIP thật sự học
python zeroshot/src/build_descriptors.py         # 167 mô tả riêng loài + bản hoán vị
python zeroshot/src/splits.py                    # 2 split seen/unseen
python zeroshot/src/encode.py --model bioclip --split test    # cache embedding
python zeroshot/src/zs_eval.py                                # bước 2 - zero-shot thuần
python zeroshot/src/train_mets.py --img bioclip --txt clip_b16 --level T0s --split variant
python zeroshot/src/train_lora.py --img bioclip --txt clip_b16 --level T0s --split variant
python zeroshot/src/build_cub.py && python zeroshot/src/train_cub.py --caps cupl
python zeroshot/src/report_zs.py --save          # gộp mọi kết quả thành 4 bảng
```

## Cấu trúc repo

```
src/                    Dự án 2 - phân loại 555 lớp có giám sát
  nabirds_io.py           port Python 3 của loader gốc + build_label_index / build_taxonomy
  prepare_images.py       pre-resize 48k ảnh (NAB_SHORT_SIDE=512 cho cache 448)
  nabirds_data.py         Dataset + augmentation FGVC + CMO + tuỳ chọn crop bbox
  models_zoo.py           factory 19 model + apply_lora() + set_num_classes()
  metrics.py              mọi chỉ số + xuất per-class / per-species / per-order / confusions
  train.py                CLI train + eval (mọi cờ augmentation)
  partquery.py            encoder-decoder: K "part query" + cross-attention
  cub_transfer.py         C3 - chuyển giao NABirds -> CUB-200-2011
  --- phân tích, không train gì ---
  report.py               gộp các run thành bảng so sánh
  aug_report.py           tách chỉ số theo lớp đuôi / lớp khó (BẮT BUỘC cho augmentation)
  ensemble.py             TTA + ensemble + độ không-trùng-lỗi + ECE
  logit_adjust.py         logit adjustment hậu kiểm cho lớp đuôi
  --- vận hành ---
  run_all.sh run_448.sh run_cnn2.sh run_more.sh run_more2.sh   pass 1-5
  run_aug.sh run_aug2.sh run_partq.sh run_cub.sh               pass 6-8
  sweep_lock.sh           khoá chống chạy chồng (PID file riêng mỗi sweep)
  status.sh stop_all.sh watch_resources.py preflight.sh run_full.sh

zeroshot/               Dự án 3 - CLIP / METS
  src/zs_env.py           thiết lập env; MỌI script phải import đầu tiên
  src/variants.py         parse 60 chuỗi biến thể -> (sex, age, season, morph, form)
  src/descriptors.py      mô tả thị giác CHUNG suy từ các trục đã parse
  src/build_descriptors.py 167 mô tả RIÊNG từng lá + bản hoán vị làm đối chứng
  src/prompts.py          10 mức prompt (bảng SPEC) + 80 template OpenAI
  src/zs_models.py        clip_b16 / bioclip / bioclip2 qua open_clip
  src/encode.py           cache embedding ảnh ra .npy
  src/zs_eval.py          đánh giá zero-shot + variant_probe
  src/splits.py           2 split seen/unseen: `variant` và `species`
  src/train_mets.py       B1 - projection head trên embedding đóng băng
  src/train_lora.py       B2 - thêm LoRA vào image tower
  src/build_cub.py, train_cub.py   B4 - InfoNCE thật trên CUB
  src/supervised_ceiling.py        trần tham chiếu từ model có giám sát
  src/report_zs.py        gộp mọi summary.json thành 4 bảng
  data/                   7 artifact CSV/JSON/NPZ (228 KB, commit được)
  results/                summary.json mỗi run + 4 bảng tổng hợp
  cache/                  embedding + HF cache (786 MB, .gitignore)

docs/                   tài liệu chi tiết (xem cuối file)
results/                summary.json + per_class.csv mỗi run, và 5 bảng tổng hợp:
                          comparison.csv|.md        mọi run
                          aug_comparison.csv|.md    tách theo lớp đuôi / lớp khó
                          logit_adjust.csv|.md      logit adjustment hậu kiểm
                          ensemble/                 ensemble + TTA
                          cub/comparison.csv        chuyển giao sang CUB
runs/                   checkpoint + log + cache logits (.gitignore, ~6 GB)
nabirds/                dataset NABirds gốc, 17 GB (.gitignore)
cub/CUB_200_2011/       CUB-200-2011 BẢN ĐẦY ĐỦ, 1.2 GB (.gitignore) - 11,788 ảnh,
                        200 lớp, split 5,994/5,794, 15 part keypoint, bbox, attribute
models/                 mọi checkpoint pretrain tải về, 5.5 GB (.gitignore)
PLAN.md                 kế hoạch, tiến độ, 19 quyết định đã chốt, việc còn lại
```

## Dữ liệu — ba điều phải biết trước khi làm gì

1. **555 lớp lá ≠ 555 loài.** Cây phân loại gộp về **404 loài**; **288/555 lá là biến thể
   giới tính / tuổi / mùa / morph** của cùng một loài (`Baltimore Oriole (Adult male)` và
   `(Female/Immature male)` là hai lớp). 48.5% ảnh test thuộc loài có nhiều hơn một lá.
2. **Cây KHÔNG đồng đều độ sâu.** 265 lá ở depth 4, 290 lá ở depth 3 → tầng depth-2
   **không** phải lúc nào cũng là family. Chỉ hai mức gộp đáng tin: **species** (cha trực
   tiếp, 404 node) và **order** (depth-1, 22 node).
3. **NABirds không có tên khoa học.** `classes.txt` ghi `Perching Birds`, không phải
   `Passeriformes`. Mọi việc dính tới BioCLIP đều phải dựng bảng ánh xạ ngoài trước.

Chi tiết: [`docs/01-dataset.md`](docs/01-dataset.md).

---

## Dự án 2 — phân loại 555 lớp, có giám sát

Giao thức chuẩn của các paper: dùng nguyên `train_test_split.txt` (23,929 / 24,633),
**không dùng bbox lúc test**, test chạy một lần với checkpoint tốt nhất theo val.
Các model pretrain đều **full fine-tune** (chỉ đóng băng backbone ở epoch 0) với 2 nhóm
learning rate, backbone thấp hơn head 10 lần.

**44 run qua 8 pass.** Kỷ lục repo: **90.03%** (`vit_b_16_in21k` @448).

Bảng dưới liệt kê **27 run kiến trúc** (cấu hình mặc định của mỗi model). 17 biến
thể augmentation nằm ở [§ Augmentation](#augmentation-top-1-tổng-che-mất-tác-dụng-thật).

| run | năm | params | pretrain | input | top-1 | top-5 | macro-F1 | @404 sp | phút |
|---|---|---|---|---|---|---|---|---|---|
| **`vit_b_16_in21k`** | 2020 | 86.7M | ImageNet-21k | 448 | **90.03** | 98.82 | 88.30 | 91.77 | 157 |
| `bioclip` | 2024 | 86.5M | **TreeOfLife-10M** | 224 | 87.72 | 98.45 | 85.09 | 89.48 | 32 |
| **part-query** (frozen) | — | 15.1M | ImageNet-1k | 224 | 86.98 | 97.83 | 85.17 | 88.72 | 12 |
| `vit_b_16_in21k` | 2020 | 86.2M | ImageNet-21k | 224 | 85.86 | 97.53 | 83.91 | 87.85 | 28 |
| `convnext_tiny_in22k` | 2022 | 28.2M | ImageNet-22k | 224 | 85.55 | 97.19 | 83.23 | 87.40 | 18 |
| `vit_b_16_in21k` +LoRA r16 | 2020 | 2.8M | ImageNet-21k | 224 | 85.06 | 96.98 | 82.88 | 86.95 | 39 |
| `vit_small_in21k` | 2020 | 21.9M | ImageNet-21k | 224 | 84.16 | 96.85 | 81.98 | 86.14 | 12 |
| `swin_tiny_in22k` | 2021 | 27.9M | ImageNet-22k | 224 | 83.78 | 96.50 | 81.28 | 85.63 | 20 |
| `efficientnetv2_s` | 2021 | 20.9M | ImageNet-21k | 224 | 83.47 | 96.11 | 81.01 | 85.43 | 35 |
| `resnet50` | 2015 | 24.6M | ImageNet-1k | 448 | 83.18 | 96.31 | 80.74 | 85.06 | 58 |
| **part-query** (ft_k24) | — | 102.2M | ImageNet-1k | 224 | 81.68 | 95.76 | 79.29 | 83.47 | 23 |
| `convnext_tiny` | 2022 | 28.2M | ImageNet-1k | 224 | 81.49 | 95.23 | 78.86 | 83.30 | 18 |
| `inception_v3` | 2015 | 25.9M | ImageNet-1k | 299 | 80.93 | 95.25 | 78.80 | 82.52 | 24 |
| `resnext50_32x4d` | 2016 | 24.1M | ImageNet-1k | 224 | 79.69 | 94.04 | 77.34 | 81.57 | 22 |
| `resnet101` | 2015 | 43.6M | ImageNet-1k | 224 | 79.56 | 94.73 | 76.97 | 81.55 | 24 |
| `vit_b_16` | 2020 | 86.2M | ImageNet-1k | 224 | 79.19 | 95.25 | 76.83 | 80.90 | 33 |
| `resnet50` | 2015 | 24.6M | ImageNet-1k | 224 | 78.61 | 94.40 | 75.86 | 80.50 | 17 |
| **part-query** (ft) | — | 101.4M | ImageNet-1k | 224 | 78.18 | 94.39 | 75.66 | 80.02 | 21 |
| `densenet121` | 2017 | 7.5M | ImageNet-1k | 224 | 77.70 | 92.98 | 75.29 | 79.49 | 21 |
| `efficientnet_b0` | 2019 | 4.7M | ImageNet-1k | 224 | 76.98 | 93.52 | 74.49 | 78.78 | 18 |
| `vgg16_bn` | 2014 | 136.5M | ImageNet-1k | 224 | 75.09 | 93.14 | 72.50 | 77.06 | 37 |
| **part-query** (imagenet) | — | 101.4M | ImageNet-1k | 224 | 75.05 | 92.86 | 72.49 | 76.80 | 21 |
| `googlenet` | 2014 | 6.2M | ImageNet-1k | 224 | 72.21 | 91.92 | 69.60 | 74.03 | 16 |
| `mobilenet_v3_large` | 2019 | 4.9M | ImageNet-1k | 224 | 68.59 | 88.27 | 65.61 | 70.60 | 15 |
| `cnn_scratch` | — | 5.0M | không | 448 | 63.74 | 85.55 | 60.08 | 65.59 | 89 |
| `cnn_scratch` | — | 5.0M | không | 224 | 60.72 | 83.85 | 57.22 | 62.49 | 26 |
| `alexnet` | 2012 | 59.3M | ImageNet-1k | 224 | 38.29 | 64.14 | 35.71 | 40.20 | 12 |

### Năm đòn bẩy, đo bằng thí nghiệm đối chứng

Mỗi dòng là một cặp run **chỉ khác nhau đúng một biến**.

| đòn bẩy | thí nghiệm đối chứng | điểm |
|---|---|---|
| Có pretrain vs không | `cnn_scratch` → `resnet50`, cùng 224 | **+17.89** |
| Nguồn pretrain IN1k → IN21k (**ViT**) | `vit_b_16` → `vit_b_16_in21k` | **+6.67** |
| **Độ phân giải 224 → 448** | `resnet50` +4.57 · `vit_b_16_in21k` **+4.17** | **+4.2…4.6** |
| Nguồn pretrain IN1k → IN22k (**CNN**) | `convnext_tiny` → `convnext_tiny_in22k` | **+4.06** |
| Nguồn pretrain IN21k → sinh học | `vit_b_16_in21k` → `bioclip` | **+1.86** |
| Kiến trúc CNN 2016 → 2022 | `resnet50`(AdamW) → `convnext_tiny`(AdamW) | +2.12 |
| **Ngân sách epoch 30 → 100** | `resnet50`, cùng mọi thứ khác | **+1.64** |
| Grouped conv | `resnet50` → `resnext50_32x4d`, gần bằng params | +1.08 |
| Độ sâu R50 → R101 | cùng pretrain, cùng 224, cùng SGD | +0.95 |
| Optimizer SGD → AdamW | `resnet50`, cùng mọi thứ khác | +0.76 |

> **Cảnh báo phải đọc kèm bảng trên.** `resnet50` chạy 100 epoch đạt **80.25**
> (best epoch **73**, không phải 23 như bản 30 epoch) — tức **mọi run CNN 30-epoch
> trong repo đang thiếu train ~1.6 điểm**. Các so sánh kiến trúc vẫn *công bằng
> nội bộ* vì đều chạy 30 epoch, nhưng con số tuyệt đối đều thấp hơn khả năng thật,
> và **ngân sách epoch là đòn bẩy lớn hơn cả độ sâu lẫn grouped conv**. Không được
> đọc "ResNet-50 chỉ đạt 78.61" như một giới hạn của kiến trúc đó.
>
> Về cơ chế: train lâu giúp **lớp đuôi** (+2.26 F1) nhưng **không giúp 20 lớp khó
> nhất** (22.70 → 22.37). Nhóm khó bị chặn bởi đặc trưng/độ phân giải chứ không
> phải bởi tối ưu hoá — chỉ độ phân giải 448 mới nâng được nó (+5.45).

Thứ tự **pretrain > độ phân giải > kiến trúc** vẫn đứng, nhưng sau 44 run phải nói
chính xác hơn: trục pretrain **trên CNN** (+4.06) và trục độ phân giải (+4.2…4.6)
**xấp xỉ nhau**; chỉ trên ViT thì pretrain mới bỏ xa (+6.67). Ở đáy bảng, **đòn bẩy
optimizer (+0.76) gần bằng đòn bẩy độ sâu (+0.95)** — `resnet50`+AdamW đạt 79.37,
chỉ kém `resnet101`+SGD (79.56) đúng 0.19 điểm.

### Độ phân giải vá được lỗ hổng lập luận lớn nhất của repo

README bản cũ *suy đoán* "khoảng cách ~5 điểm với TransFG chủ yếu là độ phân giải"
mà chưa hề đo. Nay đã đo:

| | top-1 | macro-F1 | @404 sp | F1@113 lớp đuôi | recall@20 lớp khó | phút |
|---|---|---|---|---|---|---|
| `vit_b_16_in21k` @**448** | **90.03** | **88.30** | **91.77** | **77.90** | **38.73** | 157 |
| `vit_b_16_in21k` @224 | 85.86 | 83.91 | 87.85 | 72.48 | 33.28 | 28 |
| chênh | **+4.17** | +4.39 | +3.92 | **+5.42** | **+5.45** | 5.6× |

Khoảng cách với TransFG (90.8) thu từ ~5 điểm còn **0.77** — mà TransFG dùng *đúng
backbone này, đúng độ phân giải này*, cộng thêm Part Selection Module. Suy đoán cũ
**đúng**: toàn bộ PSM của TransFG chỉ đáng ≤0.77 điểm.

Đáng chú ý hơn: độ phân giải thắng ở **đúng chỗ khó** (+5.42 F1 lớp đuôi, +5.45
recall lớp khó) — khác hẳn BioCLIP ở mục dưới. Hợp lý, vì các lớp khó là bộ lông ẩn
cần chi tiết nhỏ (vạch cánh, hoa văn đầu) mà 224px không đủ để thấy.

### Nguồn pretrain: bốn điểm trên cùng một trục

Cùng ViT-B/16, cùng SGD, cùng lr 1e-3/1e-2, cùng 25 epoch — chỉ đổi *nguồn* trọng số:

| nguồn pretrain | run | top-1 | cộng dồn |
|---|---|---|---|
| không có | `cnn_scratch` (kiến trúc khác, 5M) | 60.72 | — |
| ImageNet-1k | `vit_b_16` | 79.19 | — |
| ImageNet-21k | `vit_b_16_in21k` | 85.86 | +6.67 |
| **TreeOfLife-10M** (sinh học) | **`bioclip`** | **87.72** | +1.86 |

IN1k → sinh học = **+8.53**, vượt biên độ +5.9 mà Cui et al. CVPR'18 đo khi đổi
ImageNet → iNaturalist trên chính NABirds. **Nhưng phần lớn cái lợi đó đã lấy được
bằng IN21k rồi** (+6.67 trong 8.53): *quy mô* nguồn pretrain quan trọng hơn *tính
chuyên ngành* của nó.

**Hai cảnh báo bắt buộc đọc kèm.**

1. +1.86 của BioCLIP đến **hoàn toàn từ lớp đầu**:

   | | top-1 | F1@113 lớp đuôi | recall@20 lớp khó |
   |---|---|---|---|
   | `bioclip` | **87.72** | 69.84 | 29.49 |
   | `vit_b_16_in21k` @224 | 85.86 | **72.48** | **33.28** |

   Model "tốt nhất" chỉ tốt nhất khi tính gộp — trên lớp đuôi nó *thua* 2.6 điểm F1.

2. Rủi ro **nhiễm ở mức ảnh** (PLAN mục A2) áp cho con số này. Repo chỉ tải được
   danh sách *tên taxa* của TreeOfLife-10M (`txt_emb_species.json`, 63 MB), **không
   có ảnh**, nên chưa loại trừ được khả năng BioCLIP đã thấy chính ảnh test lúc
   pretrain — ảnh NABirds từ Flickr, EOL cũng lấy Flickr.

### CNN hiện đại có bắt kịp ViT không?

| | params | phút | top-1 | macro-F1 | @404 sp |
|---|---|---|---|---|---|
| `vit_b_16_in21k` | 86.2M | 28.3 | **85.86** | **83.91** | **87.85** |
| `convnext_tiny_in22k` | **28.2M** | **17.5** | 85.55 | 83.23 | 87.40 |
| chênh | −67% | −38% | **−0.31** | −0.68 | −0.45 |

Chênh −0.31 nhỏ hơn sai số chuẩn của val (~±0.72) → **ngang nhau về độ chính xác,
rẻ hơn hẳn về chi phí**. ConvNeXt-T IN22k cũng vượt bản LoRA của ViT (85.06): muốn
tiết kiệm thì **đổi kiến trúc hiệu quả hơn đổi cách fine-tune**, và không phải trả
cái giá "chậm hơn 38%" của LoRA.

Ở **cùng cỡ ~22–28M và cùng pretrain 21k/22k**, ba họ kiến trúc xếp hạng rõ:

| model | params | top-1 |
|---|---|---|
| `convnext_tiny_in22k` (CNN) | 28.2M | **85.55** |
| `vit_small_in21k` (ViT phẳng) | 21.9M | 84.16 |
| `swin_tiny_in22k` (ViT phân cấp) | 28.0M | 83.78 |

*Caveat:* bộ lr AdamW được probe **trên ConvNeXt** rồi dùng lại cho Swin — đúng loại
lỗi mà bẫy #12 cảnh báo, nên chênh 1.77 với Swin chưa phải kết luận sạch.

### Trục thời gian kiến trúc, cùng IN1k @224

| năm | model | params | top-1 |
|---|---|---|---|
| 2012 | `alexnet` | 59.3M | **38.29** |
| 2014 | `googlenet` | 6.2M | 72.21 |
| 2014 | `vgg16_bn` | 136.5M | 75.09 |
| 2015 | `resnet50` | 24.7M | 78.61 |
| 2016 | `resnext50_32x4d` | 24.1M | 79.69 |
| 2017 | `densenet121` | 7.5M | 77.70 |
| 2019 | `efficientnet_b0` | 4.7M | 76.98 |
| 2019 | `mobilenet_v3_large` | 4.9M | 68.59 |
| 2022 | `convnext_tiny` | 28.3M | 81.49 |

**2012 → 2022: +43.2 điểm.** Nhưng riêng 2014 → 2022 chỉ +6.4 — gần như toàn bộ
tiến bộ nằm ở bước nhảy AlexNet → GoogLeNet/VGG.

Hai điều đi ngược trực giác:

1. **Đòn bẩy pretrain CÓ giới hạn.** `alexnet` có pretrain ImageNet (38.29) **thua
   `cnn_scratch` không pretrain gì tới 22.4 điểm** (60.72). Kiểm lại đường học để
   chắc không phải lỗi lr: train-acc lên **87.84%** trong khi val đứng ở 48 — tối ưu
   hoá chạy tốt, đây là thất bại về *khái quát hoá*. Kiến trúc 2012 yếu tới mức trọng
   số pretrain không cứu nổi.
2. **Không có "trần 5M tham số".** `efficientnet_b0` (4.7M) đạt **76.98**, hơn
   `mobilenet_v3_large` (4.9M — cùng năm, cùng pretrain, cùng 224) **8.39 điểm**, lớn
   hơn cả đòn bẩy độ phân giải. MobileNetV3 tối ưu cho độ trễ di động bằng cách cắt
   mạnh số kênh ở tầng sâu — đúng nơi giữ chi tiết phân biệt loài.

Từ 4.7M lên 28.2M chỉ mua thêm **+4.51 điểm** (76.98 → 81.49): bão hoà rất sớm.

### Ba đòn bẩy KHÔNG cần train lại gì

| | chi phí | kết quả |
|---|---|---|
| **TTA** (lật ngang, + đa thang cho model tích chập) | 0 phút train | +0.2…**+1.71** |
| **Ensemble** (trung bình softmax, chọn thành viên trên val) | 0 phút train | 0.00 — xem dưới |
| **Logit adjustment** hậu kiểm (Menon et al., ICLR'21) | 0 phút train | **+5.78 F1 lớp khó** |

**TTA càng có lãi khi model càng yếu:** `efficientnetv2_s` +1.71, `mobilenet_v3_large`
+1.61, `resnext50` +1.34 — nhưng `vit_b_16_in21k` @448 **−0.04** và
`convnext_tiny_in22k` 0.00. Model mạnh đã bất biến với lật/thang sẵn rồi.

**Ensemble — kết quả âm, và lý do đáng ghi hơn con số.** Chọn thành viên tham lam
trên val, báo cáo trên test:

```
+vit_b_16_in21k_448   val 93.33 | test 90.00   (1 model)
[QUY TẮC DỪNG] thêm bioclip chỉ được +0.11 val, không vượt sai số ghép cặp ±0.30
+bioclip_224          val 93.45 | test 90.96   (2 model)  <- test đỉnh
+resnet50_448         val 93.59 | test 90.76
+efficientnetv2_s     val 93.62 | test 90.48
```

Val đạt đỉnh ở 4 model, test đạt đỉnh ở 2. Kết luận trung thực **không** phải
"ensemble vô dụng" mà là **tập val 3,510 ảnh quá nhỏ để xếp hạng ensemble** — mọi
chênh lệch ở đây (0.11–0.29 val) nằm trong nhiễu của chính nó. Con số 90.96 chỉ nhìn
thấy được *sau khi* đã xem test, nên không được tính là kết quả.

Dư địa thì có thật: `vit_b_16_in21k_448` và `bioclip` **cùng sai chỉ 6.35% số ảnh**,
tức oracle của cặp đó là **93.65** — còn 3.65 điểm cho một bộ gộp tốt hơn.

**Logit adjustment là đòn bẩy miễn phí tốt nhất còn lại.** `tau` chọn trên val,
báo cáo trên test:

| run | tau* | top-1 | Δ | F1@đuôi | Δ | F1@20 lớp khó | **Δ** |
|---|---|---|---|---|---|---|---|
| `vit_b_16_in21k` @448 | 0.25 | 90.04 | +0.15 | 77.88 | +1.05 | 45.28 | **+4.44** |
| `efficientnetv2_s` | 1.00 | 83.47 | +0.27 | 67.87 | +1.18 | 29.63 | **+5.78** |
| `bioclip` | 0.375 | 87.72 | +0.24 | 69.79 | +2.35 | 33.94 | +3.18 |

Top-1 gần như không đổi nhưng F1 nhóm khó tăng tới **+5.78**. `tau*` do val chọn
phản ánh model thiên vị lớp phổ biến tới mức nào — model đã cân bằng sẵn được val
chọn `tau = 0`, tức **tự động không áp dụng gì**, nên không có nguy cơ làm hỏng.

### Augmentation: top-1 tổng che mất tác dụng thật

**18 run**, mỗi run đổi **đúng một biến**. Chỉ số phải tách theo nhóm — 113 lớp
đuôi chỉ chiếm ~10% ảnh test nên top-1 tổng che mất hiệu ứng thật (**D17**).

| `resnet50` @224 | top-1 | F1@113 lớp đuôi | recall@20 lớp khó |
|---|---|---|---|
| baseline (30 ep) | 78.61 | 62.04 | 22.70 |
| `--rotate 15` | 78.50 **(−0.11)** | **63.29 (+1.25)** | **25.43 (+2.73)** |
| `--hue 0.10` | **77.92 (−0.69)** | **61.91 (−0.13)** | 23.40 |
| `--erasing-p 0` | 78.69 (+0.08) | 63.18 | 24.41 |
| 100 epoch | **80.25 (+1.64)** | **64.30 (+2.26)** | 22.37 (−0.33) |
| 100 epoch + CutMix | **80.48** | 63.87 | 24.11 **(+1.74)** |

**1. Xoay 15° trông vô dụng trên top-1** (−0.11) nhưng nâng nhóm đuôi +1.25 và
nhóm khó +2.73. Nếu chỉ theo dõi top-1 thì nó đã bị bỏ đi — đây là lý do
`src/aug_report.py` tồn tại.

**2. Quyết định D3 được xác nhận bằng số.** Tăng hue lên 0.10 làm hại **gấp 2.4
lần** ở 77 lớp mà màu là nhãn (Warbler/Tanager/Oriole/Finch/Bunting: −1.22 F1)
so với 478 lớp còn lại (−0.50). Đúng cơ chế D3 dự đoán, không phải trùng hợp.
`hue 0.10` cũng là phép biến đổi **duy nhất làm hại cả nhóm đuôi**.

**3. Tắt `RandomErasing`: thật nhưng rất nhỏ, và chỉ ở top-1.** Chạy đủ **bốn**
model:

| model | Δ top-1 | Δ F1@đuôi | Δ recall@khó |
|---|---|---|---|
| `resnet50` | +0.08 | +1.14 | +1.71 |
| `convnext_tiny_in22k` | +0.13 | −0.49 | −0.45 |
| `vit_b_16_in21k` | **+0.33** | +0.61 | −2.35 |
| `bioclip` | +0.16 | +1.51 | +2.00 |

**Top-1 dương ở 4/4 model** (+0.08…+0.33, trung bình +0.175). Bốn lần cùng dấu
thì xác suất ngẫu nhiên chỉ 1/16 ⇒ **hiệu ứng thật, nhưng nhỏ hơn nhiễu của từng
run riêng lẻ**. Nhóm đuôi/khó mới là chỗ **không** nhất quán (3/4 và 2/4).

Tài liệu ghi Cutout hại FGVC ~2 điểm; `RandomErasing(scale=0.02–0.15)` của repo
nhẹ hơn hẳn Cutout chuẩn nên chỉ tạo ra ~0.18 điểm. Bài học phương pháp: sau MỘT
model tôi tưởng đã xác nhận (+1.14 F1 đuôi); sau HAI model tôi kết luận "không
nhất quán"; phải đủ **bốn** mới thấy đúng bản chất — nhất quán ở top-1, nhiễu ở
các chỉ số nhóm nhỏ.

**4. Ngân sách epoch là đòn bẩy lớn hơn kiến trúc** — +1.64 so với +1.08 (grouped
conv) và +0.95 (nhân đôi độ sâu).

**5. CutMix dương chứ không âm — nhưng nhỏ.** So với baseline 100 epoch riêng:
**+0.23** top-1, trong khi benchmark OpenMixup đo **+2.67** trên CUB. Lý do đáng
ghi: **NABirds có 23,929 ảnh train, gấp ~4 lần CUB (5,994)** — CutMix là bộ điều
chuẩn nên giá trị co lại khi dữ liệu dồi dào. Ghi chú cũ của repo ("mixup thường
**hại** hơn lợi") **sai về dấu**.

#### CMO: lợi ích tỉ lệ NGHỊCH với độ mạnh của model

CMO (Park et al., CVPR 2022) dán chim của 113 lớp hiếm (pool **2,377 ảnh**) lên
nền của lớp nhiều ảnh — lớp thiểu số thiếu *bối cảnh đa dạng* nên mượn của lớp đa
số. Bản của repo dùng **bbox thật** thay vì ô ngẫu nhiên như bản gốc.

Trên `resnet50` (model **yếu**) — có tác dụng và có điểm tối ưu rõ:

| `cmo-p` | top-1 | F1@đuôi | recall@khó |
|---|---|---|---|
| 0 | 78.69 | 63.18 | 24.41 |
| 0.25 | **78.73** | 64.50 | 25.80 |
| 0.50 | 78.25 | **64.71** | **28.28** |
| 0.75 | 77.36 | 63.31 | 25.87 |

Đơn điệu tăng tới 0.50 rồi quay đầu ⇒ **quan hệ liều–đáp ứng thật**, không phải
nhiễu. Ở 0.50 nó cho **+5.58 recall lớp khó** so với baseline gốc.

Trên model **mạnh** (cmo-p 0.5, so với chính bản `_noerase` của model đó) — **hại**:

| model | Δ top-1 | Δ F1@đuôi | Δ recall@khó |
|---|---|---|---|
| `convnext_tiny_in22k` | −0.44 | +0.61 | −0.16 |
| `vit_b_16_in21k` | −1.00 | −1.48 | −1.72 |
| `bioclip` | **−1.17** | **−5.51** | **−10.98** |

> **Kết luận quan trọng nhất của phần augmentation.** Giá trị của CMO **tỉ lệ
> nghịch với độ mạnh của model** — đúng dạng đã thấy ở TTA. Model yếu được lợi từ
> bối cảnh đa dạng thêm; model mạnh vốn đã xử lý được lớp đuôi nên ảnh ghép chỉ
> còn là **nhiễu nhãn** mà nó đủ sức khớp vào và hỏng theo. BioCLIP hỏng nặng
> nhất (−10.98 recall lớp khó), hợp lý vì nó có đặc trưng chuyên biệt nhất.
>
> Và các phép augmentation này **đạp nhau chứ không cộng dồn**: `_noerase +
> rot15 + cmo50` cho 63.33 F1 đuôi, **thấp hơn CMO 0.50 một mình** (64.71).

Con số tuyệt đối của bộ ba trên ba model mạnh:

| model | mặc định | `_noerase` | `_noerase + CMO 0.5` |
|---|---|---|---|
| `bioclip` | 87.72 | **87.88** | 86.71 |
| `vit_b_16_in21k` | 85.86 | **86.19** | 85.19 |
| `convnext_tiny_in22k` | 85.55 | **85.68** | 85.24 |

`bioclip_224_noerase` **87.88** là model đơn tốt thứ hai của repo, chỉ sau
`vit_b_16_in21k` @448 (90.03).

**Đã bỏ có bằng chứng (D16):** augmentation sinh ảnh bằng diffusion. SaSPA
(NeurIPS 2024) đo trên 5 dataset thì **CUB được lợi ít nhất (+0.7** so với
CompCars +5.7), và chính tác giả giải thích là *class-fidelity thấp* — model sinh
ảnh không giữ được đặc điểm loài khi các loài quá giống nhau. Cần 4× RTX 3090 để
đổi lấy mức lợi nhỏ nhất, kèm rủi ro sinh ảnh SAI loài.

Chi tiết + nguồn: [`docs/02` § Pass 6](docs/02-supervised.md).

### Encoder-decoder: part-query decoder cho kết quả dương mạnh nhất

Kiến trúc: encoder = backbone **pretrain sẵn**, decoder = K "part query" học được
+ self-attention + cross-attention vào token của encoder (15.14M tham số).

Khác INTR (ICLR 2024) ở một điểm quyết định: **query gắn với BỘ PHẬN, không gắn
với LỚP**. INTR dùng C query (một cho mỗi lớp) và chính tác giả nêu hạn chế là
tốn khi C > số ô lưới N — NABirds có **C = 555 > N = 196**, rơi đúng vùng xấu.
Ở đây K = 12 ≪ 196.

| run | encoder | top-1 | F1@đuôi | recall@khó |
|---|---|---|---|---|
| **`partq_frozen`** | ViT-B/16 IN21k đã fine-tune NABirds, **ĐÓNG BĂNG** | **86.98** | **74.78** | **37.51** |
| *(encoder của chính nó)* | `vit_b_16_in21k_224` | 85.86 | 72.48 | 33.28 |
| **chênh** | | **+1.12** | **+2.30** | **+4.23** |
| `partq_ft_k24` | như trên, mở khoá, K=24 | 81.68 | 66.74 | 26.80 |
| `partq_ft` | như trên, mở khoá, K=12 | 78.18 | 62.73 | 22.12 |
| `partq_imagenet` | ImageNet (đối chứng kiểu INTR) | 75.05 | — | — |

1. **Chỉ train 15.14M tham số decoder trên encoder ĐÓNG BĂNG cho +1.12 top-1 và
   +4.23 recall lớp khó** — mức tăng lớn nhất mà một module thêm vào đạt được
   trong repo, và không đụng gì tới trọng số encoder.
2. **Nhánh fine-tune HỎNG** (78.18 so với 86.98). Nguyên nhân gần như chắc chắn:
   AdamW lr 1e-4 phá đặc trưng của ViT vốn được train bằng SGD — đúng **bẫy #12**.
   Lỗi kỹ thuật sửa được, không phải thất bại của ý tưởng.
3. **Đối chứng kiểu INTR đúng như dự đoán**: khởi tạo từ ImageNet cho 75.05, kém
   bản đóng băng **11.93 điểm**. Khớp với việc INTR chỉ đạt CUB 71.8% so với
   83.8% của ResNet-50 thường — vấn đề là backbone yếu, không phải kiến trúc.

Khảo sát đầy đủ (HERBS, INTR, PDiscoFormer, Saccadic Vision):
[`docs/05`](docs/05-attention-encoder-decoder.md).

### NABirds → CUB: chuyển giao dương nhưng nhỏ

Cùng recipe, cùng 25 epoch, khác **đúng một biến** là trọng số khởi tạo backbone:

| model | ImageNet → CUB | ImageNet → **NABirds** → CUB | chênh |
|---|---|---|---|
| `bioclip` | 90.21 | **90.49** | **+0.28** |
| `vit_b_16_in21k` | 89.85 | **90.02** | +0.17 |
| `convnext_tiny_in22k` | 89.47 | **89.78** | +0.31 |

**3/3 model đều dương**, trung bình +0.25 — hướng nhất quán nên không phải nhiễu,
nhưng biên độ nhỏ. Đúng như dự đoán trước khi chạy: **142/200 loài CUB đã có mặt
trong NABirds**, nên đây không phải transfer sang miền mới mà chỉ là "thấy thêm
ảnh chim". Cui et al. đo +5.9 khi đổi ImageNet → iNaturalist vì iNat lớn hơn
NABirds hai bậc độ lớn.

### LoRA có thay được full fine-tune không?

Cùng pipeline, cùng val split, cùng 25 epoch; chỉ khác optimizer (AdamW thay SGD, vì SGD
lr 0.001 gần như không đẩy được adapter). LoRA gắn vào `qkv`, `attn.proj`, `fc1`, `fc2`.

| | tham số train | top-1 | macro-F1 | @404 sp | **var81bal** | phút |
|---|---|---|---|---|---|---|
| full fine-tune | 86.2M (100%) | **85.86** | **83.91** | **87.85** | 95.61 | **28.3** |
| LoRA r=16 | **2.79M (3.2%)** | 85.06 | 82.88 | 86.95 | 95.58 | 39.1 |
| chênh | −96.8% | **−0.80** | −1.04 | −0.90 | **−0.03** | +38% |

Ba điều đọc được:

1. **LoRA giữ được 99.1% độ chính xác với 3.2% tham số train.** Với bài này, chênh 0.80
   điểm là cái giá phải trả.
2. **Trên trục biến thể thì hai bên trùng khít** (95.58 vs 95.61). Khả năng phân biệt bộ
   lông được LoRA phục hồi trọn vẹn; 0.80 điểm mất đi nằm ở chỗ khác — phân biệt loài với
   loài.
3. **LoRA CHẬM HƠN 38% về thời gian thực** (39.1 vs 28.3 phút), dù train ít tham số hơn
   30 lần. Adapter thêm matmul mỗi bước forward, backward vẫn phải lan qua toàn mạng, và
   AdamW nặng hơn SGD. **Ít tham số train ≠ train nhanh hơn.**

Chi tiết (recipe, thiết kế val, phân tích lỗi, nhật ký sự cố):
[`docs/02-supervised.md`](docs/02-supervised.md).

---

## Dự án 3 — zero-shot kiểu CLIP / METS

### Tiền đề bị dữ liệu bác bỏ

Ý tưởng ban đầu là zero-shot **ở mức loài** với BioCLIP. Bước kiểm tra đầu tiên đã giết
nó: đối chiếu 404 loài NABirds với danh sách taxa mà BioCLIP đã embed
(`imageomics/TreeOfLife-10M`, 384,490 taxa) cho **403/404 = 99.8% trùng ở mức loài**.

> **Không được gọi kết quả loài trên NABirds là "zero-shot".** Đó là đánh giá
> **in-domain đã pretrain**.

Trục còn hợp lệ là **biến thể bộ lông**: BioCLIP được giám sát *hoàn toàn bằng
taxonomy*, nhãn giới tính / tuổi / mùa / morph không tồn tại trong nguồn train của nó.

> **Thuật ngữ dùng thống nhất từ đây (quyết định D11).** Trục biến thể là
> **compositional zero-shot**, KHÔNG phải ZSL cổ điển: model đã **thấy ảnh** chim mái
> khi pretrain, nó chỉ chưa thấy **nhãn** "female". ZSL cổ điển đòi lớp unseen không
> xuất hiện dưới bất kỳ dạng nào. Gọi tắt là "zero-shot" ở trục này là chỗ dễ bị phản
> biện đúng nhất, nên mọi bảng và kết luận bên dưới đều ghi rõ *compositional*.

### Phép đo trung tâm

`var81bal` — cho sẵn đúng loài, model có chọn đúng biến thể không? Thu hẹp không gian
nhãn về đúng các lá của loài đó, **lấy trung bình accuracy của từng lá** (cân bằng, nên
thiên vị một biến thể tự triệt tiêu). Tính trên 81 loài của split `variant`.

Thang đo đầy đủ — mọi con số dưới đây phải đọc theo thang này:

```
chance                                     47.57
BioCLIP off-the-shelf (prompt tốt nhất)    62.96   ── phục hồi 32% khoảng cách
CLIP off-the-shelf (prompt tốt nhất)       79.90   ── phục hồi 67%
METS-analogue (clip+clip, T0s)             85.63   ── phục hồi 79%
model CÓ GIÁM SÁT (trần thực nghiệm)       95.61   ── 100%
```

Trần lấy từ `vit_b_16_in21k` của dự án 2 — model đã thấy đủ 555 nhãn nên nó là mức trên
đạt được trên chính dataset này (`zeroshot/src/supervised_ceiling.py`). Sanity check:
script tính lại top-1 555 lớp ra 85.89 so với 85.86 đã ghi trong bảng dự án 2.

### Kết quả chính: định danh loài và đọc bộ lông là hai năng lực **tách rời**

Cùng một prompt (`T0s`) cho cả ba model:

| model | kiến trúc | loài 404-way | **var81bal** |
|---|---|---|---|
| CLIP ViT-B/16 | 86M | 50.87 | **79.90** |
| BioCLIP ViT-B/16 | 86M | 81.36 | 62.96 |
| BioCLIP-2 ViT-L/14 | 304M | **95.66** | 64.02 |

So sánh sạch nhất là **CLIP vs BioCLIP** (cùng ViT-B/16, chỉ khác nguồn giám sát):
đổi sang BioCLIP được **+30.5 điểm** định danh loài nhưng mất **−16.9 điểm** đọc bộ lông.
Giám sát bằng taxonomy mua độ chính xác phân loại học bằng cái giá là khả năng nối ngôn
ngữ mô tả với ảnh. Kết quả này giữ nguyên ở cả ba prompt cố định khác nhau (`T0s`/`T2`/`T3b`).

Và: **BioCLIP-2 — không train gì — vượt model có giám sát của chính repo này 7.8 điểm ở mức loài**
(95.66 vs 87.85) mà không thấy một ảnh NABirds nào — thước đo mức độ nhiễm, không phải
bằng chứng tổng quát hoá.

### Thang mô tả text (var81bal)

| | chỉ tên loài | +cụm biến thể | +mô tả chung | +mô tả riêng loài |
|---|---|---|---|---|
| **CLIP** | 48.50 | 75.32 | 76.35 | **79.90** |
| BioCLIP | 48.50 | 50.04 | 59.21 | 62.66 |
| BioCLIP-2 | 48.50 | 58.08 | 56.29 | 63.98 |

Mọi cấu hình không mang thông tin biến thể đều cho **48.50 ≈ chance 47.57** — kiểm tra
tự nhất quán đạt.

**Đối chứng: mô tả riêng loài có thật sự mang thông tin, hay chỉ cần text đa dạng?**
Hoán vị 167 mô tả giữa các loài, giữ nguyên vai trò male/female (`Northern Cardinal
(Adult Male)` nhận mô tả của một loài trống khác):

| model | mô tả CHUNG | mô tả ĐÚNG | mô tả **HOÁN VỊ** |
|---|---|---|---|
| CLIP | 76.35 | **79.90** | 66.68 |
| BioCLIP | 59.21 | **62.66** | 54.57 |
| BioCLIP-2 | 56.29 | **63.98** | 52.67 |

Mô tả hoán vị tụt **9.4–13.2 điểm** so với mô tả đúng, và **thấp hơn cả mô tả chung** —
mô tả sai-nhưng-cụ-thể còn hại hơn không mô tả gì. Nội dung mô tả mang thông tin thật;
kết luận "mô tả riêng loài > mô tả chung" đứng vững.

### METS-analogue: train projection head, text tower đóng băng

Split `variant` — 81 lá female/immature **không có trong train**:

| image tower | text tower | var81bal sau train | không train | chênh |
|---|---|---|---|---|
| clip_b16 | clip_b16 | **85.63** | 79.90 | +5.73 |
| **bioclip** | **clip_b16** | **77.66** | — | **+13.51 so với BioCLIP thuần** |
| bioclip2 | bioclip2 | 68.02 | 60.40 | +7.62 |
| bioclip | bioclip | 64.15 | 54.86 | +9.29 |

Train giúp trục biến thể ở **mọi** cấu hình, dù 81 lá đó chưa từng xuất hiện. **Ghép
ảnh-BioCLIP với text-CLIP ăn thua** — đúng như dự đoán từ bảng tách rời ở trên.

Split `species` (79 loài giữ khỏi train): train có lãi +1.3 đến +6.8 điểm ZSL ở mọi model.

**Độ ổn định (3 seed, var81bal):**

| split | cấu hình | trung bình ± độ lệch chuẩn |
|---|---|---|
| variant | clip+clip/T0s | 85.40 ± 0.29 |
| variant | bioclip+clip/T0s | 76.53 ± 1.60 |
| variant | bioclip+bioclip/T2 | 63.48 ± 1.64 |
| species | bioclip+bioclip/T2 | 87.90 ± 0.80 |
| species | bioclip2+bioclip2/T2 | 81.19 ± **6.73** ⚠ |

Chênh lệch "ghép hai tower" là **+13.05 ± 2.3** → vững. Riêng `bioclip2` trên split
`species` dao động ±6.73, mọi con số của ô đó phải đọc kèm cảnh báo này.

### LoRA trên image tower (B2)

| split | cấu hình | LoRA | B1 (linear) | all555 LoRA / B1 | phút | VRAM |
|---|---|---|---|---|---|---|
| variant | bioclip+clip/T0s | 77.36 | 77.66 | 69.36 / 63.67 | 12.7 | 5.4 GB |
| variant | bioclip+bioclip/T2 | 65.69 | 64.15 | 73.46 / 72.43 | 12.1 | 5.4 GB |
| variant | bioclip2+bioclip2/T2 | 72.83 | 68.02 | **87.54** / 84.59 | 27.6 | 10.0 GB |
| species | bioclip+clip/T0s | 91.06 | 87.84 | 68.91 / 62.56 | 11.9 | 5.4 GB |
| species | bioclip+bioclip/T2 | 90.87 | 87.29 | 76.38 / 72.39 | 12.0 | 5.4 GB |

**LoRA gần như không thêm gì cho trục biến thể (compositional zero-shot)** (+1.5 / −0.3 trên split
variant) nhưng nâng rõ độ chính xác tổng thể. Nghĩa là **nút thắt không nằm ở đặc trưng
ảnh** — thông tin bộ lông đã có sẵn trong feature đóng băng — mà ở phía text và phép nối.

### Độ hạt của text có quan trọng không? (B4, trên CUB)

Cùng 5,994 ảnh CUB, đổi đúng một biến là độ hạt caption:

| caption | số chuỗi duy nhất | val-unseen CUB | zsl352 NABirds |
|---|---|---|---|
| **CuPL** (biến thiên theo ảnh) | 4,490 | **73.73** | **41.54** |
| naive (mức lớp) | 200 | 68.95 | 39.46 |

Text biến thiên thắng ở **cả ba learning rate và cả ba chỉ số** — bằng chứng trực tiếp
rằng độ hạt text là biến có thật. Nhưng train trên 6k ảnh CUB **không thắng nổi**
off-the-shelf (41.54 vs BioCLIP-2 77.95): giá trị của B4 nằm ở phép so sánh có kiểm soát.

Chi tiết phương pháp: [`docs/03-zeroshot-method.md`](docs/03-zeroshot-method.md).
Kết quả đầy đủ: [`docs/04-zeroshot-results.md`](docs/04-zeroshot-results.md).

---

## Bài học kỹ thuật — 18 cái bẫy đã gặp

Phần này là thứ đáng giá nhất khi chạy lại repo trên máy khác.

**Nạp model / thư viện**

| # | Bẫy | Hậu quả đo được | Ở đâu |
|---|---|---|---|
| 1 | `ViT-B-16` + tag `openai` dựng **sai activation** (CLIP gốc dùng QuickGELU) | CLIP bị hạ oan **6.6 điểm**; chỉ có một `UserWarning` lẫn trong log | [docs/04 §12.1](docs/04-zeroshot-results.md) |
| 2 | open_clip gộp q/k/v vào `nn.MultiheadAttention` (`in_proj_weight`) | `peft` **không** gắn LoRA vào q/k/v được; chỉ còn `c_fc`/`c_proj`/`out_proj` | [docs/03 §9](docs/03-zeroshot-method.md) |
| 3 | `peft` khớp `target_modules` theo **hậu tố**, nên `proj` trúng cả `patch_embed.proj` | LoRA gắn nhầm vào Conv2d nhúng patch | `src/models_zoo.py` |
| 4 | `import peft` kéo theo `tensorflow` build cho NumPy 1.x | Python chết ngay; sửa bằng `USE_TF=0` trong `zs_env.py` | [docs/04 §10.5](docs/04-zeroshot-results.md) |
| 5 | timm ViT có `qkv` là **một** `nn.Linear` — ngược lại với (2) | cùng kiến trúc, hai thư viện, hai khả năng adapt | `src/models_zoo.py` |

**Dữ liệu / tra cứu ngoài**

| # | Bẫy | Hậu quả | Ở đâu |
|---|---|---|---|
| 6 | GBIF chứa **loài hoá thạch** trùng tên thường (`Aquila bivia` = "Golden Eagle") | sai tên khoa học; lọc bằng `extinct` | [docs/04 §11.2](docs/04-zeroshot-results.md) |
| 7 | GBIF trả **bản ghi lai** không có `canonicalName` (`Anas rubripes x platyrhynchos`) | cột tên khoa học rỗng **mà vẫn báo khớp** | như trên |
| 8 | Tên thường trùng giữa Tân/Cựu Thế giới (`Black Vulture` → `Aegypius monachus`) | trả về loài châu Âu cho dataset Bắc Mỹ | như trên |
| 9 | TreeOfLife-10M dùng **taxonomy cũ hơn** GBIF (`Dendroica` vs `Setophaga`) | đưa tên GBIF cho BioCLIP là đo nhầm — chuỗi ngoài phân phối | [docs/04 §11.1](docs/04-zeroshot-results.md) |

**Phương pháp đo**

| # | Bẫy | Hậu quả | Ở đâu |
|---|---|---|---|
| 10 | Chỉ số biến thể tính **chỉ trên lá female** thì gian lận được | model thiên vị "female" ăn 85.68 trong khi thực tế 54.89 | [docs/04 §16](docs/04-zeroshot-results.md) |
| 11 | Chọn epoch bằng **val-seen** trong bài zero-shot | ZSL-unseen 62.59 trong khi không train gì được 69.70 | [docs/04 §13](docs/04-zeroshot-results.md) |
| 12 | Bê nguyên mức lr SGD từ ViT sang **ConvNeXt** | ConvNeXt kẹt ở val **32.54** sau 4 epoch; AdamW cùng điều kiện đạt **81.91** — chênh **49 điểm**, mà loss vẫn giảm đều nên không có dấu hiệu lỗi nào. Không phải chuyện LayerNorm: ViT dùng đúng mức lr đó vẫn đạt 86.84 | [docs/02 § Pass 3](docs/02-supervised.md) |
| 13 | So model mới (AdamW) với baseline cũ (SGD) rồi gọi chênh lệch là "kiến trúc" | +0.76 trong +2.88 là do **optimizer**, không phải kiến trúc. Phải chạy thêm baseline bằng đúng optimizer mới để bóc ra | như trên |
| 16 | `inception_v3` / `googlenet`: `transform_input` mặc định **False** khi `weights=None` nhưng bị torchvision ép **True** khi có weights | dựng khung bằng `pretrained=False` rồi nạp checkpoint đã train ⇒ **inception mất 6.06 điểm, googlenet mất 7.11 điểm**, `load_state_dict` vẫn thành công vì đó là thuộc tính bool KHÔNG nằm trong state_dict — sai hoàn toàn lặng lẽ. Đã thêm chốt chặn trong `src/ensemble.py`: cache logits mà lệch >0.5 điểm so với `summary.json` thì raise ngay | `src/models_zoo.py` |
| 18 | **Sửa file nguồn trong lúc một sweep đang chạy** | Trên Windows, DataLoader worker được **spawn**: nó import lại module (code MỚI) rồi unpickle object dataset (dựng bằng code CŨ). Object cũ thiếu thuộc tính mới ⇒ `AttributeError` trong worker, **giết run ở giữa** sau khi đã train xong. Mất một run vì đúng lỗi này. Đọc thuộc tính bằng `getattr(self, 'x', mặc_định)` chỉ là băng dán — cách đúng là **không sửa `src/` khi sweep đang chạy** | `src/nabirds_data.py` |
| 17 | Chọn thành viên ensemble tham lam trên tập val 3,510 ảnh | val tăng đều 93.33→93.62 trong khi **test đi xuống** 90.96→90.48. Chênh lệch 0.11–0.29 val nằm trong nhiễu của chính nó. Đã thêm ngưỡng theo **sai số ghép cặp McNemar** | `src/ensemble.py` |

**Vận hành trên Windows**

| # | Bẫy | Hậu quả | Ở đâu |
|---|---|---|---|
| 14 | DataLoader không đặt `persistent_workers` | Windows spawn lại worker mỗi epoch, GPU tụt xuống **3%**; sửa xong nhanh **2.5x** | [docs/02](docs/02-supervised.md) |
| 15 | `torch.compile` không dùng được trên Windows; `tee` che tiến độ; giết task không giết bash con | xem nhật ký sự cố | [docs/02](docs/02-supervised.md) |

---

## Môi trường

| | |
|---|---|
| GPU | RTX 5070 12GB (Blackwell sm_120) — dùng **bf16**, không dùng fp16 |
| Python | 3.12.10 |
| Chính | `torch 2.12.0.dev+cu128`, `torchvision`, `timm 1.0.29` |
| Zero-shot | `open_clip_torch 3.3.0`, `peft 0.20.0`, `transformers 4.51.3` |
| Khác | `pandas`, `scikit-learn`, `pyarrow`, `huggingface_hub` |

Mọi checkpoint pretrain tải về `models/` (`TORCH_HOME` / `HF_HOME` được set trong code),
nên không rải ra ngoài repo.

## Tài liệu chi tiết

| file | nội dung |
|---|---|
| [`PLAN.md`](PLAN.md) | kế hoạch, tiến độ, quyết định đã chốt, việc còn lại |
| [`docs/01-dataset.md`](docs/01-dataset.md) | cấu trúc NABirds, taxonomy, bbox/keypoint, kết quả EDA |
| [`docs/02-supervised.md`](docs/02-supervised.md) | dự án 2: nghiên cứu, recipe, thiết kế val, phân tích lỗi, nhật ký sự cố |
| [`docs/03-zeroshot-method.md`](docs/03-zeroshot-method.md) | dự án 3: vì sao đổi trục, vì sao đây là DeViSE chứ không phải CLIP, thiết kế prompt |
| [`docs/04-zeroshot-results.md`](docs/04-zeroshot-results.md) | dự án 3: toàn bộ kết quả bước 1→4, hạn chế |
| [`docs/05-attention-encoder-decoder.md`](docs/05-attention-encoder-decoder.md) | khảo sát HERBS / INTR / PDiscoFormer / Saccadic Vision + đề xuất decoder "part query" cho repo này |
| [`EDA_NABirds.ipynb`](EDA_NABirds.ipynb) | notebook EDA (đã chạy, có sẵn biểu đồ) |
| `results/comparison.md`, `zeroshot/results/tables.md` | bảng kết quả sinh tự động |

## Tham khảo

- Cui et al., *Large Scale Fine-Grained Categorization and Domain-Specific Transfer
  Learning*, CVPR 2018 — arXiv:1806.06193
- Stevens et al., *BioCLIP: A Vision Foundation Model for the Tree of Life*, CVPR 2024 —
  arXiv:2311.18803, HF `imageomics/bioclip`
- Li et al., *Frozen Language Model Helps ECG Zero-Shot Learning* (METS), MIDL 2023 —
  arXiv:2303.12311
- Menon & Vondrick, *Visual Classification via Description from Large Language Models*,
  ICLR 2023 (CuPL/DCLIP)
- He et al., *TransFG*, AAAI 2022 — arXiv:2103.07976; Diao et al., *MetaFormer*,
  arXiv:2203.02751
- Xian et al., *Zero-Shot Learning — A Comprehensive Evaluation of the Good, the Bad and
  the Ugly*, TPAMI 2019 (giao thức GZSL và cách chọn epoch)
