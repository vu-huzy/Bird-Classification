# 2. Bài toán 555 lớp có giám sát — kế hoạch, triển khai, kết quả

> Tách từ README gốc. Gồm cả nhật ký sự cố vận hành — phần đáng đọc nhất
> nếu bạn định chạy lại trên máy khác.

> **Ghi chú điều hướng:** "mục 0"–"mục 6" là các mục đánh số trong chính file này.

# Nghiên cứu & Kế hoạch huấn luyện (2026-09-06)

Phần này ghi lại kết quả **đọc data + tra cứu paper**, chưa có code. Mục tiêu: chốt kiến trúc,
độ phân giải, cách crop, checkpoint pretrain và giao thức zero-shot trước khi viết dòng code đầu tiên.

## 0. Số liệu mới đo được trên chính dataset (bổ sung cho phần EDA)

| Đại lượng | Giá trị | Ý nghĩa quyết định |
|---|---|---|
| bbox cạnh dài — median | **500 px** (p5=233, p95=834) | Crop bbox rồi resize 224 là **thu nhỏ** với 95.7% ảnh → không có blur do upscale |
| bbox cạnh dài < 224px | **4.34%** ảnh | Crop → 224 gần như luôn an toàn |
| bbox cạnh dài < 448px | 38.8% ảnh (chỉ 4.3% phải phóng >2x) | Crop → 448 vẫn chấp nhận được |
| bbox chiếm diện tích | median **28.3%**, mean 30.7% | Ở ảnh đầy đủ 224x224, con chim chỉ chiếm ~**124 px** cạnh hiệu dụng |
| Ảnh gốc | median 1024x683; 0.09% có cạnh ngắn <224 | Decode ảnh gốc là bottleneck → nên pre-resize offline |
| Ảnh/lớp (train) | mean 43.1, min 4, max **60** (bị chặn trần), p5=17 | Imbalance **nhẹ**, không cần weighted sampler phức tạp |
| Lớp có <30 ảnh train | 113 / 555 (39 lớp <20, 7 lớp <10) | Few-shot tail thật sự, nhưng không cực đoan |
| Node phân loại | 555 lá → **404 species** → **22 order** | Cây KHÔNG đồng đều: 265 lá ở depth 4 (`order > family > species > biến thể`), 290 lá ở depth 3 (`order > species > biến thể`, bỏ qua family). Vì vậy tầng depth-2 **không** phải lúc nào cũng là family (199/228 node depth-2 thực chất là species) → chỉ dùng species (cha trực tiếp) và order (depth-1) làm mức gộp |
| Lớp lá là biến thể giới tính/tuổi/bộ lông | **288 / 555** (có ngoặc đơn trong tên) | ~52% nhầm lẫn 555-way là **nhầm trong cùng loài** |
| Keypoint hữu dụng | mỏ 98.6%, đỉnh đầu 97.7%, mắt chỉ ~52%, TB 8.6/11 part hiện | Part-based model khả thi, nhưng mắt/cánh thiếu nhiều |
| Leakage (lớp, photographer) | **0.4% ảnh** | Split gốc sạch — dùng thẳng |

**Chồng lấn CUB-200-2011 và NABirds (đã kiểm chứng, không phải phỏng đoán):**

Đối chiếu 200 tên lớp CUB với 404 species NABirds (khớp token + xử lý tay 11 alias:
Cardinal→Northern Cardinal, Mockingbird→Northern Mockingbird, White_Pelican→American White Pelican,
Tree_Sparrow→American Tree Sparrow, Florida_Jay→Florida Scrub-Jay, Forsters_Tern→Forster's Tern,
Nighthawk→Common Nighthawk, Myrtle_Warbler→Yellow-rumped Warbler, Geococcyx→Greater Roadrunner,
Great_Grey_Shrike→Northern Shrike, Sayornis→Eastern Phoebe):

| | Số lượng |
|---|---|
| Species CUB **có** trong NABirds | **142 / 200 (71.0%)** |
| Species CUB **không** có (hải âu, auklet, jaeger, kingfisher nhiệt đới, sparrow hiếm...) | 58 |
| Lớp lá NABirds thuộc một species của CUB (**seen**) | **203 / 555 (36.6%)** — 19,249 ảnh |
| Lớp lá NABirds **không** có trong CUB (**unseen sạch**) | **352 / 555 (63.4%)** — 29,313 ảnh (**14,907 ảnh test**) |

→ Đây là **seen/unseen split hợp lệ** cho thí nghiệm zero-shot CUB → NABirds. Lưu ý trung thực khi
viết báo cáo: unseen ở **mức loài**, không phải mức họ (nhiều lớp unseen vẫn cùng genus/family với
lớp seen). Muốn khó hơn thì loại thêm ở mức family (depth-2).

## 1. Bài toán CNN: input/output, crop, augmentation

### Chốt về crop (quan trọng nhất)

- **Không crop bbox làm setting chính.** Toàn bộ SOTA trên NABirds (TransFG, MetaFormer, MPSA...) đều
  báo cáo **không dùng bbox lúc test**. Dùng GT bbox ở test là setting khác, không so sánh được.
- **Nhưng crop bbox là thí nghiệm đối chứng bắt buộc** vì số liệu ở trên cho thấy nó tăng độ phân giải
  hiệu dụng của con chim từ ~124px lên 224px (gần **1.9x** theo cạnh) mà **không** phải phóng to ảnh.
- Cách crop đúng: nới bbox thêm **12–20% mỗi chiều** (giữ bối cảnh cành/nước — có tín hiệu phân biệt),
  kẹp vào biên ảnh (201 bbox tràn biên), rồi resize.
- Nếu muốn crop mà vẫn "hợp lệ" ở test: train một attention/detector tự sinh box (đó chính là ý tưởng
  của WS-DAN / MMAL-Net), không dùng GT box.

### Pipeline chuẩn

```
TRAIN                                    TEST
RandomResizedCrop(R, scale=(0.3,1.0))    Resize(R*1.14, cạnh ngắn)
RandomHorizontalFlip(0.5)                CenterCrop(R)
ColorJitter(0.2, 0.2, 0.1, hue<=0.02)    Normalize(ImageNet mean/std)
Normalize(ImageNet mean/std)
RandomErasing(p=0.25)   # optional
```

- `scale` mặc định của `RandomResizedCrop` là `(0.08,1.0)` — **quá mạnh cho FGVC**, sẽ cắt mất con chim.
  Bắt buộc đổi thành `(0.3,1.0)` hoặc `(0.5,1.0)`.
- **Không** vertical flip, không xoay quá +-10 độ (chim có hướng chuẩn).
- **Hue jitter phải rất nhỏ**: màu bộ lông chính là nhãn (Yellow vs Orange Warbler).
- `RandomErasing` đang bật mặc định (`p=0.25`) — **chưa ai kiểm trên dataset này**, và tài liệu
  ghi Cutout làm ResNet-50 **mất ~2 điểm** trên CUB vì có xác suất xoá trúng vùng phân biệt.
  Đây là ablation rẻ nhất còn lại (PLAN mục E1).
- Mixup/CutMix: chỉ bật khi train >40 epoch, và dùng nhẹ (alpha 0.2). Với FGVC thường **hại** hơn lợi.
  > **Đính chính (2026-09-07, sau khi tra benchmark).** Câu "thường hại hơn lợi" **quá mạnh**.
  > Benchmark OpenMixup trên CUB-200 (transfer learning, **200 epoch**) cho CutMix **+2.67**
  > và SAMix **+3.82** trên ResNeXt-50 so với vanilla; chỉ FMix là tụt (−0.40 trên ResNet-18).
  > Phần đúng của ghi chú gốc là điều kiện **số epoch** — repo chạy 25–30 epoch nên chưa ở vùng
  > mixup phát huy. Hệ quả: **chạy mixup ở 25 epoch rồi kết luận "vô dụng" là lỗi đo.**
  > Xem PLAN.md mục E.

### Input/Output

| Dự án | Input | Output | Loss |
|---|---|---|---|
| #1 Binary | `(B,3,224,224)` | `(B,1)` logit | `BCEWithLogitsLoss` |
| #2 Multi-class | `(B,3,R,R)`, R thuộc {224,384,448} | `(B,555)` logits | `CrossEntropy(label_smoothing=0.1)` |
| #3 Zero-shot | `(B,3,224,224)` + text | 2 vector 512-d đã L2-norm | InfoNCE đối xứng, temperature học được |

**Nhãn binary nên chọn:** `Perching Birds (Passeriformes)` vs phần còn lại — 24,787 vs 23,775 ảnh,
cân bằng gần hoàn hảo, dùng được **toàn bộ 48k ảnh**. Hướng thú vị hơn (nếu muốn có "câu chuyện"):
`waterbird vs landbird` gộp theo order — tái hiện benchmark **Waterbirds** về spurious correlation
(model học nền nước thay vì con chim); có bbox để chứng minh điều đó bằng cách so accuracy
ảnh-đầy-đủ vs ảnh-crop.

### CNN "tay tự xây" cho dự án #1 (baseline để thấy khoảng cách với pretrain)

```
Input  (B,3,224,224)
Stem   Conv3x3 3->32 s2 + BN + ReLU                 -> 112x112
Blk1   [Conv3x3 32->64  + BN + ReLU] x2 + MaxPool2  -> 56x56
Blk2   [Conv3x3 64->128 + BN + ReLU] x2 + MaxPool2  -> 28x28
Blk3   [Conv3x3 128->256+ BN + ReLU] x2 + MaxPool2  -> 14x14
Blk4   [Conv3x3 256->512+ BN + ReLU] x2 + MaxPool2  -> 7x7
Head   GlobalAvgPool -> (B,512) -> Dropout(0.3) -> Linear(512->C)
~5M params
```

Kỳ vọng: binary ~85–90%; nếu ép chạy 555 lớp thì chỉ ~20–35% → đúng mục đích minh hoạ.

### Metric nên báo cáo (khai thác cấu trúc dataset)

- Top-1 / Top-5 trên **555 lớp lá**
- **Macro-F1** (vì 113 lớp có <30 ảnh train)
- **Accuracy mức species (404 lớp)** = gộp các biến thể giới tính/tuổi. Chênh lệch giữa 555-way và
  404-way cho biết bao nhiêu lỗi là "nhầm giới tính/bộ lông cùng loài" — phân tích có giá trị riêng
  vì 288/555 lớp là biến thể.
- **Accuracy mức order (22 lớp)** — sai ở mức này là sai nghiêm trọng.

## 2. Biến thể CNN nâng cao — nên chạy cái nào

Bảng SOTA đã tra được trên **NABirds** (top-1, không dùng bbox lúc test):

| Method | Backbone | Res | Pretrain | NABirds | CUB |
|---|---|---|---|---|---|
| ResNet-50 fine-tune (tham chiếu) | R50 | 448 | IN1k | ~82–85% | ~85% |
| Cui et al. CVPR'18 | Inception-v3 | 560 | **iNaturalist** | **87.91%** | **89.26%** |
| Cui et al. CVPR'18 | Inception-v3 | 560 | ImageNet | 82.01% | 82.84% |
| TransFG (AAAI'22) | ViT-B/16 | 448 | IN21k | 90.8% | 91.7% |
| M2Former | ViT-B/16 | 448 | IN21k | 91.1% | — |
| GLSim | ViT-B/16 | 448 | IN21k | 92.0% | — |
| MPSA | ViT | 448 | IN21k | 92.5% | — |
| **MetaFormer-0** | MetaFormer-0 (28M) | 384 | **iNat2021** | **91.5%** | 91.8% |
| MetaFormer-0 | MetaFormer-0 (28M) | 224 | IN21k | 89.5% | 89.7% |
| MetaFormer-2 | MetaFormer-2 (81M) | 384 | iNat2021 | **93.0%** | 92.9% |
| Token Injection Transformer | ViT | 448 | IN21k | 93.2% | — |

**Kết luận rút ra:** nguồn pretrain đóng góp **~6 điểm** (82.0 → 87.9 khi đổi ImageNet → iNaturalist),
lớn hơn hầu hết cải tiến kiến trúc. Ưu tiên đúng thứ tự: **pretrain > độ phân giải > kiến trúc**.

**Với RTX 5070 12GB, nên chạy (theo thứ tự):**

1. **PMG** (Progressive Multi-Granularity, ECCV'20) — jigsaw + train tăng dần độ chi tiết, chỉ dùng
   1 backbone ResNet-50, gần như không thêm tham số. CUB 89.6%. **Rẻ nhất / lợi nhất.**
2. **WS-DAN** (Weakly Supervised Data Augmentation) — sinh attention map rồi attention-crop /
   attention-drop. CUB 89.4%. Đây cũng là cách "crop hợp lệ" mà không cần GT bbox → nối thẳng vào
   phân tích crop ở mục 1.
3. **Bilinear CNN / Compact Bilinear Pooling** — pooling bậc hai, kinh điển của FGVC (CUB 84.1%).
   Rẻ, dễ giải thích, hợp làm phần "CNN nâng cao" trong báo cáo.
4. **DCL** (Destruction and Construction, CVPR'19) — CUB 87.8% với ResNet-50.

**Không nên đuổi theo:** TransFG (tốn VRAM vì giữ attention toàn bộ 12 layer), MetaFormer-2 @384
(cần iNat21 checkpoint 81M + nhiều GPU). MetaFormer-**0** @224 IN21k (28M) thì hoàn toàn khả thi.

## 3. ResNet / ViT-B: bao nhiêu lớp, pretrain gì, train ra sao

### Bao nhiêu lớp?

**Đừng tự thiết kế độ sâu.** Chỉ các cấu hình chuẩn mới có trọng số pretrain, mà pretrain đáng giá
~6 điểm accuracy. Cụ thể:

- **ResNet-50** = bottleneck blocks `[3,4,6,3]`, 25.6M params, output 2048-d. Chuẩn de-facto của FGVC.
  ResNet-18/34 chỉ dùng làm baseline chạy nhanh.
- **ViT-B/16** = **12 layer**, dim 768, 12 head, patch 16x16. Ở 224 → 196 token; ở 448 → **784 token**
  (attention tốn ~16x so với 224).
- Thay đổi duy nhất là **head**:
  - ResNet: `GlobalAvgPool -> (BN) -> Dropout(0.2) -> Linear(2048, 555)`. Nâng cấp nhỏ: đổi GAP thành
    **GeM pooling** (+0.5–1% cho FGVC).
  - ViT: `LayerNorm -> Linear(768, 555)`. Nâng cấp nhỏ: nối `CLS` với `mean(patch tokens)` →
    `Linear(1536, 555)`.

### Có nên pretrain? — **Có, bắt buộc.** Không bàn cãi với 24k ảnh train.

### Checkpoint nào, lấy ở đâu

Cần cài trước: `pip install timm open_clip_torch` (máy hiện **chưa có** cả hai; `torch 2.12+cu128`
và `torchvision` đã có và nhận đúng RTX 5070, sm_120, 11.9GB).

| Mục đích | Model ID | Nguồn |
|---|---|---|
| ResNet-50 baseline | `torchvision.models.resnet50(weights="IMAGENET1K_V2")` | torchvision (đã cài) |
| ResNet-50 recipe tốt hơn | `resnet50.a1_in1k` | timm |
| CNN hiện đại | `convnext_tiny.in12k_ft_in1k`, `convnext_small.in12k_ft_in1k` | timm |
| **ViT-B/16 IN21k (khuyến nghị chính)** | `vit_base_patch16_224.augreg_in21k` | timm — đúng checkpoint TransFG/ViT-FGVC dùng |
| ViT-B/16 @384 | `vit_base_patch16_384.augreg_in21k_ft_in1k` | timm |
| Self-supervised mạnh | `vit_base_patch14_dinov2.lvd142m` | timm |
| Rất mạnh, vừa 12GB | `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` | timm |
| **Domain-specific (chim/sinh vật)** | `imageomics/bioclip` (ViT-B/16, TreeOfLife-10M) | HuggingFace |
| **Domain-specific (iNat2021)** | MetaFormer-0/1/2 checkpoints | github.com/dqshuai/MetaFormer (Google Drive / Baidu) |

timm tự nội suy positional embedding khi đổi độ phân giải:
`timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=555, img_size=448)`.

### Công thức train (ước lượng cho 12GB, bf16 autocast — Blackwell nên dùng bf16, không dùng fp16)

| Cấu hình | batch | VRAM (ước lượng) | Ghi chú |
|---|---|---|---|
| ResNet-50 @224 | 64 | ~6–7 GB | thoải mái |
| ResNet-50 @448 | 24–32 | ~9–10 GB | |
| ViT-B/16 @224 | 64 | ~8 GB | |
| ViT-B/16 @384 | 16 (+grad accum) | ~9 GB | bật `set_grad_checkpointing()` |
| ViT-B/16 @448 | 8 (+grad accum -> 32) | ~10 GB | bắt buộc grad checkpointing |

- **ResNet-50:** SGD momentum 0.9, wd 1e-4, lr backbone 1e-3 / head 1e-2, cosine, warmup 500 step,
  **30–40 epoch**, label_smoothing 0.1.
- **ViT-B/16:** AdamW lr 1e-4, wd 0.05, **layer-wise lr decay 0.65**, cosine, warmup 500 step,
  **10–20 epoch là đủ** (ViT-B trên FGVC hội tụ nhanh, train lâu sẽ overfit). Bản TransFG dùng
  SGD lr 0.03 — cũng chạy được, nhưng AdamW + LLRD ổn định hơn.
- **Hai giai đoạn** (khuyến nghị): (1) đóng băng backbone, train head 2–3 epoch; (2) mở toàn bộ,
  cosine đến 0. Tránh việc head khởi tạo ngẫu nhiên phá trọng số pretrain ở vài step đầu.
- **EMA** trọng số: +~0.3%.
- **Không dùng weighted sampler.** Imbalance ở đây là 60/4 nhưng phân bố chính nằm trong 32–60 ảnh;
  oversampling lớp 4-ảnh sẽ overfit. Nếu muốn xử lý tail: Class-Balanced loss (Cui et al. 2019) hoặc
  chỉ cần label smoothing.

### Tối ưu tốc độ (Windows)

- **Pre-resize offline** toàn bộ 48k ảnh về cạnh ngắn 512–600 px, lưu JPEG q=92. Ảnh gốc median
  1024x683, decode là bottleneck thật sự — bước này thường rút ngắn epoch time nhiều hơn mọi tinh
  chỉnh khác. Chỉ 0.09% ảnh có cạnh ngắn <224 nên không mất thông tin cho R<=448.
- `DataLoader(num_workers=8, persistent_workers=True, pin_memory=True, prefetch_factor=4)`.
- `channels_last` + `torch.autocast('cuda', bfloat16)`. `torch.compile` là tuỳ chọn (trên Windows
  đôi khi không ổn định).

### Kỳ vọng kết quả trên NABirds 555 lớp (ước lượng, dựa trên bảng SOTA ở trên)

| Cấu hình | Top-1 kỳ vọng |
|---|---|
| CNN tự xây, from scratch @224 | ~~20–35%~~ → **đo thực tế 60.72%** (xem đính chính bên dưới) |
| ResNet-50 IN1k @224 | 76–80% |
| ResNet-50 IN1k @448 | 82–85% |
| ConvNeXt-T IN12k @384 | 87–89% |
| **ViT-B/16 IN21k @384** | **88–89%** |
| ViT-B/16 IN21k @448 | 89–90% |
| + PMG / WS-DAN trên ResNet-50 | +1–2% so với baseline tương ứng |
| MetaFormer-0 iNat21 @384 (nếu tải được ckpt) | ~91.5% (đã công bố) |

## 4. Hướng METS (zero-shot với frozen LLM) và các hướng khác

### METS thực sự làm gì (đã xác minh — MIDL 2023, "Frozen Language Model Helps ECG Zero-Shot Learning")

- Encoder tín hiệu **ResNet1d-18 (train được)** + **language model đóng băng hoàn toàn** làm text encoder.
- Mỗi nhánh có **một linear projection head** (head này train được).
- Ghép cặp (ECG, báo cáo lâm sàng **sinh tự động bằng máy**) → contrastive kiểu CLIP.
- Zero-shot: embed tên/mô tả lớp mới bằng **cùng** LM đóng băng → nearest neighbour trong không gian chung.
- Test zero-shot trên MIT-BIH (tập lớp hoàn toàn khác) → +~10% so với baseline có giám sát.

### Bản sao cho chim

```
Ảnh  --[ViT-B/16 hoặc ResNet-50, TRAIN ĐƯỢC]--> 768-d --[Linear, train]--> 512-d --L2norm--\
                                                                                            >-- InfoNCE đối xứng
Text --[LLM ĐÓNG BĂNG]------------------------> 768/1024-d --[Linear, train]--> 512-d --L2norm--/
```

**Mẹo then chốt:** vì text encoder đóng băng, **precompute toàn bộ text embedding một lần**
(CUB có ~118k caption → vài phút), rồi cache. Lúc train không tốn thêm VRAM cho nhánh text, batch
size dồn hết cho nhánh ảnh. Đây là điều làm cho toàn bộ hướng này vừa 12GB.

### Nguồn text: NABirds **không có caption** — xử lý thế nào

| Nguồn | CUB | NABirds |
|---|---|---|
| Caption theo từng ảnh (Reed et al. 2016, 10 câu/ảnh) | có | **không** |
| 312 attribute nhị phân | có | **không** |
| Tên lớp | có | có |
| Đường đi taxonomy (order → family → species → biến thể) | hạn chế | **4 tầng đầy đủ** |

→ Dùng **mô tả mức lớp sinh bằng LLM** (đúng tinh thần METS: báo cáo của METS cũng là *máy sinh*,
không phải người viết). Với mỗi lớp, sinh 5–10 câu mô tả hình thái kiểu Menon & Vondrick / CuPL:
*"Yellow-rumped Warbler (Breeding Myrtle): a small songbird with a bright yellow rump patch, yellow
side patches, a black mask, and white wing bars."* Cộng thêm chuỗi taxonomy phẳng kiểu BioCLIP:
`"Aves Passeriformes Parulidae Setophaga coronata"`.

### Frozen LLM nào — đề xuất chạy 3 biến thể để có ablation

| Vai trò | Model | Chiều | Ghi chú |
|---|---|---|---|
| Baseline | `openai/clip-vit-base-patch16` (text tower) | 512 | mốc so sánh bắt buộc |
| **Domain-matched (mạnh nhất cho chim)** | **`imageomics/bioclip`** text tower | 512 | ViT-B/16 + text transformer, train trên TreeOfLife-10M (454k taxa). Zero-shot Birds-525: **72.1%** vs CLIP 49.9%, OpenCLIP 54.7% |
| **Frozen LLM đúng nghĩa (giống METS nhất)** | `BAAI/bge-base-en-v1.5` (110M) hoặc `Qwen3-Embedding-0.6B` | 768 / 1024 | embedder dựa trên LLM, đóng băng, chỉ train linear head. `Qwen3-Embedding-0.6B` fp16 ~1.2GB → thừa chỗ trên 12GB |
| Thay thế nhẹ | `sentence-transformers/all-mpnet-base-v2` | 768 | rẻ, ổn định |

Nếu muốn bám sát METS tối đa: đóng băng hẳn một chat LLM nhỏ (`Qwen2.5-1.5B` / `Llama-3.2-1B`),
mean-pool hidden state cuối. Thường **kém hơn** embedder chuyên dụng, nhưng là ablation hợp lệ và
đúng nghĩa "frozen LLM".

### Giao thức zero-shot — đã có split hợp lệ

- **Train:** CUB-200-2011, 200 lớp, 5,994 ảnh train, caption theo ảnh.
- **Test zero-shot:** **352 lớp lá NABirds không thuộc species nào của CUB** → **14,907 ảnh test**
  (xem mục 0). Đây là câu trả lời cho câu hỏi còn treo: overlap là **142/200 species**, và nếu không
  lọc thì **39.6% ảnh NABirds** thuộc species mà model đã thấy → con số "zero-shot" sẽ bị thổi phồng.
- **Báo cáo song song 3 con số:** (a) 555 lớp đầy đủ, (b) 203 lớp seen, (c) **352 lớp unseen** ← con số
  thật sự có ý nghĩa.
- **Cảnh báo trung thực bắt buộc ghi vào báo cáo:** nếu image encoder khởi tạo từ CLIP/BioCLIP thì nó
  **đã thấy ảnh chim trên web** → không phải zero-shot thuần. Chạy cả hai: (i) init từ ImageNet-21k
  (sạch hơn), (ii) init từ CLIP. METS cũng vướng đúng vấn đề này với LM đóng băng của họ.

### Các hướng khác đáng làm (ngoài METS)

1. **Baseline zero-shot không train gì** — CLIP / OpenCLIP / **BioCLIP** / SigLIP + prompt ensembling,
   chạy thẳng trên 352 lớp unseen. Rẻ, và cho mốc tham chiếu. **Quan trọng:** model contrastive tự
   train trên 6k ảnh CUB gần như chắc chắn **thua** BioCLIP (đã học 10M ảnh). Framing đúng phải là
   *"liệu alignment nhỏ, chuyên biệt có tổng quát hoá sang loài chưa thấy không"*, **không phải**
   *"tôi đánh bại BioCLIP"*.
2. **DCLIP / CuPL** — thay prompt `"a photo of a {tên loài}"` bằng tập descriptor sinh bởi LLM rồi
   lấy trung bình. Thường +2–5% zero-shot, gần như miễn phí, và là một phân tích đẹp.
3. **Semantic side-info ablation** — so 3 loại "class embedding": (a) chỉ tên lớp, (b) chuỗi taxonomy
   phẳng kiểu BioCLIP, (c) mô tả hình thái do LLM sinh. NABirds có cây 4 tầng đầy đủ nên (b) chạy được
   trực tiếp — đây là phần **thay thế cho attribute vector mà NABirds thiếu**.
4. **Generalized ZSL (GZSL)** — test trên cả seen và unseen, báo cáo **harmonic mean**. Đây mới là giao
   thức chuẩn (Xian et al.) và khó phản biện hơn ZSL thuần.
5. **Few-shot / linear probe** trên lớp unseen (1/5/10-shot) — nối liền zero-shot với có giám sát đầy đủ.

### Về hướng transfer CUB → NABirds

Số liệu overlap ở mục 0 làm thay đổi kết luận: **39.6% ảnh NABirds thuộc species có trong CUB**, nên
"CUB → NABirds" không phải nghiên cứu domain transfer sạch — một phần là *học lại chính lớp đó*.
Cộng thêm việc NABirds (23,929 ảnh train) lớn gấp **~4x** CUB (5,994), chiều này gần như chắc chắn
cho lợi ích nhỏ.

**Đề xuất đảo chiều: pretrain trên NABirds → fine-tune CUB.** Đúng chiều "nguồn lớn → đích nhỏ" mà
Cui et al. CVPR'18 đã chứng minh (iNat → CUB: 89.26% vs ImageNet 82.84%), rẻ (chỉ thay lớp FC cuối
2048→200), và có kỳ vọng cải thiện thật. Cả hai chiều đều phải **bỏ và khởi tạo lại FC cuối** vì
CUB có 200 nhãn còn NABirds có 555 nhãn khác hẳn.

## 5. Thứ tự chạy đề xuất

```
1.  Pre-resize 48k ảnh -> cạnh ngắn 512      -> verify: epoch time giảm rõ rệt
2.  CNN tự xây, binary Passeriformes         -> verify: acc > 85%, pipeline chạy đúng
3.  ResNet-50 IN1k @224, 555 lớp             -> verify: top-1 76-80%   [BASELINE CHỐT]
4.  Lặp lại (3) với crop bbox nới 15%        -> verify: đo chênh lệch do độ phân giải con chim
5.  ResNet-50 @448                           -> verify: top-1 82-85%
6.  ViT-B/16 augreg_in21k @384               -> verify: top-1 88-89%
7.  PMG hoặc WS-DAN trên ResNet-50           -> verify: +1-2% so với (5)
8.  NABirds -> fine-tune CUB                 -> verify: so với ImageNet -> CUB trực tiếp
9.  Zero-shot: BioCLIP off-the-shelf trên 352 lớp unseen   -> mốc tham chiếu
10. METS-analogue: train CUB, test 352 lớp unseen, 3 text tower (CLIP / BioCLIP / bge)
```

## 6. Tham khảo

- Cui et al., *Large Scale Fine-Grained Categorization and Domain-Specific Transfer Learning*,
  CVPR 2018 — arXiv:1806.06193, code: github.com/richardaecn/cvpr18-inaturalist-transfer
- Diao et al., *MetaFormer: A Unified Meta Framework for Fine-Grained Recognition*, arXiv:2203.02751 —
  model zoo: github.com/dqshuai/MetaFormer
- He et al., *TransFG: A Transformer Architecture for Fine-grained Recognition*, AAAI 2022 — arXiv:2103.07976
- Stevens et al., *BioCLIP: A Vision Foundation Model for the Tree of Life*, CVPR 2024 (Oral, Best
  Student Paper) — arXiv:2311.18803, HF: `imageomics/bioclip`
- Li et al., *Frozen Language Model Helps ECG Zero-Shot Learning* (METS), MIDL 2023 — arXiv:2303.12311
- Menon & Vondrick, *Visual Classification via Description from Large Language Models*, ICLR 2023
- Du et al., *Fine-Grained Visual Classification via Progressive Multi-Granularity Training* (PMG), ECCV 2020
- Hu & Qi, *See Better Before Looking Closer: Weakly Supervised Data Augmentation Network* (WS-DAN), 2019

---

# Triển khai: phân loại 555 lớp (giao thức như các paper)

## Cấu trúc code

| File | Vai trò |
|---|---|
| [src/nabirds_io.py](src/nabirds_io.py) | Port Python 3 của `nabirds/nabirds.py` (bản gốc Python 2). Giữ nguyên tên hàm để đối chiếu; sửa `map()` trả iterator, thêm `encoding='utf-8'`. Bổ sung `build_label_index` (555 class_id → 0..554) và `build_taxonomy` (lá → species/order) |
| [src/prepare_images.py](src/prepare_images.py) | Pre-resize 48,562 ảnh về cạnh ngắn 448 (chạy 1 lần) |
| [src/nabirds_data.py](src/nabirds_data.py) | `Dataset` 555 lớp + augmentation FGVC + tuỳ chọn crop bbox |
| [src/models_zoo.py](src/models_zoo.py) | Factory 6 model, tải weights về `models/` |
| [src/metrics.py](src/metrics.py) | Toàn bộ chỉ số + xuất CSV per-class / per-order / per-species / confusions |
| [src/train.py](src/train.py) | CLI train + eval |
| [src/report.py](src/report.py) | Gộp kết quả các run thành bảng so sánh |
| [src/run_all.sh](src/run_all.sh) | Chạy tuần tự toàn bộ sweep |

## Chuẩn bị (chạy 1 lần)

```bash
pip install timm
python src/prepare_images.py
```

Bước resize giảm `nabirds/images` từ **9.4 GB → 3.3 GB** (`nabirds/images_r448`).
Lý do: ảnh gốc median 1024×683, JPEG decode là bottleneck thật. Đo được trên máy này:

| DataLoader | throughput | epoch (23,929 ảnh) |
|---|---|---|
| workers=0 | 143 img/s | 166 s |
| workers=4 | 612 img/s | 39 s |
| workers=8 | 1,124 img/s | 21 s |
| workers=12 | 1,383 img/s | 17 s |

Chỉ 0.09% ảnh có cạnh ngắn <224 nên resize về 448 không mất thông tin cho mọi
độ phân giải train ≤ 448.

## Chạy

```bash
python src/train.py --model resnet50 --epochs 15 --batch-size 64
python src/train.py --model inception_v3 --epochs 15 --batch-size 48
python src/train.py --model vit_b_16_in21k --epochs 12 --batch-size 64
bash src/run_all.sh          # chạy hết
python src/report.py --save  # bảng so sánh
python src/report.py --run resnet50_224   # chi tiết 1 run
```

`--model` nhận: `cnn_scratch`, `resnet50`, `resnet101`, `vit_b_16`,
`vit_b_16_in21k`, `inception_v3`.
Cờ khác: `--use-bbox` (thí nghiệm đối chứng crop bbox), `--no-pretrained`,
`--img-size`, `--freeze-epochs`, `--eval-only --ckpt ...`.

## Model zoo — weights tải về `models/`

`models_zoo.py` set `TORCH_HOME` và `HF_HOME` trỏ vào `models/` trước khi import
torchvision, nên mọi checkpoint nằm gọn trong repo:

```
models/hub/checkpoints/resnet50-11ad3fa6.pth          (97.8 MB)
models/hub/checkpoints/resnet101-cd907fc2.pth
models/hub/checkpoints/inception_v3_google-0cc3c7bd.pth
models/hub/checkpoints/vit_b_16-c867db91.pth
models/hf/hub/models--timm--vit_base_patch16_224.augreg_in21k/...
```

| `--model` | Backbone | Pretrain | Input | Params | Head thay thế |
|---|---|---|---|---|---|
| `cnn_scratch` | VGG-style 5 block tự xây | không | 224 | 5.0M | `Linear(512, 555)` |
| `resnet50` | ResNet-50 `[3,4,6,3]` | ImageNet-1k V2 | 224 | 24.6M | `fc: Linear(2048, 555)` |
| `resnet101` | ResNet-101 `[3,4,23,3]` | ImageNet-1k V2 | 224 | 43.6M | `fc: Linear(2048, 555)` |
| `vit_b_16` | ViT-B/16, 12 layer, dim 768 | ImageNet-1k | 224 | 86.2M | `heads.head: Linear(768, 555)` |
| `vit_b_16_in21k` | ViT-B/16 (timm `augreg_in21k`) | **ImageNet-21k** | 224 | 86.2M | `head: Linear(768, 555)` |
| `inception_v3` | Inception-v3 | ImageNet-1k | **299** | 25.9M | `fc` + `AuxLogits.fc` |

Ghi chú `inception_v3`: torchvision ép `transform_input=True` khi load weights
pretrain (weights port từ TensorFlow) nên dataset vẫn dùng chuẩn hoá ImageNet
bình thường. Lúc train, model trả `InceptionOutputs(logits, aux_logits)` →
loss = `CE(logits) + 0.4 * CE(aux_logits)` đúng recipe gốc.

## Điểm quan trọng trong pipeline dữ liệu

1. **Split**: dùng nguyên `train_test_split.txt` (23,929 / 24,633). Đã verify
   0 image_id trùng giữa 2 split.
2. **Nhãn**: 555 `class_id` lá → 0..554 (sắp theo `class_id` tăng dần).
   Verify: label range 0..554, đúng 555 giá trị phân biệt, 404 species, 22 order.
3. **`RandomResizedCrop(scale=(0.3, 1.0))`** thay vì mặc định `(0.08, 1.0)`.
   Mặc định cắt mất con chim — bbox chỉ chiếm median 28.3% diện tích ảnh.
4. **hue jitter = 0.02** (gần như tắt). Màu bộ lông chính là nhãn.
5. **Không dùng bbox lúc test** — giống TransFG / MetaFormer / MPSA. `--use-bbox`
   chỉ để chạy đối chứng.
6. **bbox được scale lại** theo tỉ lệ `width_resized / width_gốc` vì
   `bounding_boxes.txt` ghi toạ độ trên ảnh gốc, và kẹp vào biên (201 bbox tràn biên).

## Recipe train

- SGD momentum 0.9 nesterov, cosine schedule, warmup 300 step, grad clip 5.0
- 2 nhóm lr: backbone thấp, head cao (mặc định 0.005 / 0.05; ViT 0.001 / 0.01)
- Bỏ weight decay cho bias và các tham số 1 chiều (norm)
- `label_smoothing=0.1`
- `--freeze-epochs 1`: epoch đầu đóng băng backbone, chỉ train head — tránh việc
  head khởi tạo ngẫu nhiên phá trọng số pretrain ở các step đầu
- `bfloat16` autocast + `channels_last` (RTX 5070 là Blackwell sm_120, dùng bf16
  chứ không dùng fp16)
- **Không** dùng weighted sampler: imbalance ở đây nhẹ (train mean 43.1, phần lớn
  nằm trong 32–60 ảnh/lớp), oversampling các lớp 4 ảnh sẽ overfit

## Chỉ số xuất ra

Mỗi run ghi vào `results/<run_name>/`:

| File | Nội dung |
|---|---|
| `summary.json` | top-1/top-5, balanced accuracy, Cohen kappa, macro/micro/weighted P-R-F1, accuracy gộp ở mức 404 species và 22 order, tỉ lệ lỗi trong cùng loài / cùng order |
| `per_class.csv` | **555 dòng**: support, TP/FP/FN, precision, recall, F1, accuracy, top-5 recall, lớp bị nhầm sang nhiều nhất |
| `per_species.csv` | 404 dòng gộp theo loài |
| `per_order.csv` | 22 dòng gộp theo bộ |
| `confusions.csv` | 60 cặp (true, pred) bị nhầm nhiều nhất, kèm cờ cùng loài / cùng order |

**Lưu ý về micro:** bài này là single-label multiclass nên
`micro-P = micro-R = micro-F1 = top-1 accuracy` (mỗi mẫu đóng góp đúng 1 dự đoán).
Con số đáng đọc để đánh giá đuôi dài là **macro-F1** và **balanced accuracy**,
vì 113/555 lớp có <30 ảnh train.

## Giải thích chi tiết: mỗi cột kết quả được tính như thế nào

Phần "Chỉ số xuất ra" ở trên nói *có gì*; phần này nói **công thức và cơ chế
tính từng cột**, kèm số liệu thật để kiểm chứng — viết vì `top1`/`top5` bị dùng
cho hai đại lượng khác nhau ở hai chỗ khác nhau trong output, dễ gây nhầm.

### Top-1 / Top-5 — tính trên TOÀN BỘ tập test, cho ra 1 con số/model

Code thật (`src/train.py::evaluate`):

```python
out = model(x)                         # (batch, 555) điểm số thô cho 555 lớp
top5 = out.topk(5, dim=1).indices      # 5 chỉ số lớp cao nhất, xếp hạng 1->5
pred = top5[:, 0]                      # hạng 1 = dự đoán top-1
hit5 = (top5 == y[:, None]).any(1)     # nhãn thật có nằm trong 5 chỉ số đó?
```

Ví dụ: ảnh thật là "Northern Cardinal" (chỉ số lớp 300). Model xếp hạng 5 chỉ số
cao nhất là `[145, 300, 88, 302, 12]`.
- **top-1** = hạng 1 = 145 (không phải Cardinal) → **sai**.
- **top-5** = 300 có nằm trong 5 chỉ số đó không → có, ở hạng 2 → **đúng**.

`top1_accuracy = (số ảnh đúng top-1) / (tổng số ảnh test)`, `top5_accuracy`
tương tự với điều kiện lỏng hơn. Cả hai là **một con số duy nhất cho cả model**
(gộp toàn bộ 24,633 ảnh test).

**Vì sao dùng cả hai:** top-1 là con số thực dụng nếu hệ thống chỉ hiển thị 1 kết
quả. Top-5 tách biệt hai loại lỗi khác hẳn nhau: "gần đúng nhưng chưa chắc chắn
thứ hạng 1" (top-1 sai, top-5 đúng) và "không có tín hiệu gì đúng" (cả hai đều
sai). Với 288/555 lớp chỉ khác nhau ở biến thể giới tính/tuổi, top-5 cao trong
khi top-1 thấp là dấu hiệu model học được đặc trưng loài nhưng chưa đủ tinh để
phân biệt biến thể — top-1 một mình không cho biết điều đó.

### `top5_recall` trong `per_class.csv` KHÔNG PHẢI cùng con số với `top5_accuracy`

Đây là chỗ dễ nhầm nhất: cả hai cùng đo "nhãn đúng có lọt top-5 không", nhưng
trên hai tập mẫu khác nhau.

| | `top5_accuracy` (bảng so sánh, `summary.json`) | `top5_recall` (mỗi dòng `per_class.csv`) |
|---|---|---|
| Tính trên | toàn bộ 24,633 ảnh test cùng lúc | chỉ ảnh của **riêng 1 lớp** — 555 con số khác nhau |
| Code | `correct5 / n` | `top5_correct[y_true == c].mean()` |

Kiểm chứng bằng số liệu thật của `vit_b_16_in21k`:

```
top5_accuracy (global)                                      = 97.53%
TB có trọng số theo support của 555 giá trị top5_recall      = 97.53%   <- khớp global
TB KHÔNG trọng số (mỗi lớp nặng ngang nhau) của top5_recall  = 97.10%   <- thấp hơn
```

`top5_accuracy` = trung bình có trọng số (lớp đông ảnh lấn át) — dùng để **so
sánh model với nhau**. `top5_recall` trong `per_class.csv` là con số riêng từng
lớp — dùng để **tìm lớp model yếu**, không dùng để so model.

### `acc@404sp`, `acc@22ord` — gộp NHÃN trước khi so sánh, tính trên toàn bộ ảnh

Code thật (`src/metrics.py::_agg`):

```python
def _agg(y_true, y_pred, groups):
    g = np.asarray(groups)                 # groups[c] = tên loài/bộ của lớp lá c
    return accuracy_score(g[y_true], g[y_pred])
```

Với **mỗi ảnh** trong 24,633 ảnh test: tra tên loài (hoặc bộ) của nhãn thật, tra
tên loài của nhãn model đoán, so **hai tên đó** — không so trực tiếp hai chỉ số
lớp lá nữa.

Ví dụ thật, từ `confusions.csv` của `vit_b_16_in21k`:

> Nhãn thật: lớp lá **"Yellow-rumped Warbler (Breeding Audubon's)"**.
> Model đoán: lớp lá **"Yellow-rumped Warbler (Winter/juvenile Audubon's)"**.
> Ở mức 555 lớp: hai chỉ số khác nhau -> **sai**.
> Cả hai gộp về loài **"Yellow-rumped Warbler"** -> **đúng** ở `acc@404sp`.

Vì gộp nhãn chỉ có thể biến một cặp *sai* (mức mịn) thành *đúng* (mức thô hơn),
không bao giờ ngược lại, nên luôn có: `acc@22ord >= acc@404sp >= top1`.

### `err_same_species%`, `err_same_order%` — khác `acc@404sp` ở MẪU SỐ

Code thật:

```python
wrong = y_true != y_pred                     # LỌC TRƯỚC: chỉ giữ ảnh đã sai ở mức lá
errors_within_same_species_pct = mean(species[y_true[wrong]] == species[y_pred[wrong]]) * 100
```

| | `acc@404sp` | `err_same_species%` |
|---|---|---|
| Mẫu số | **toàn bộ** 24,633 ảnh test | **chỉ** ảnh đã sai ở mức 555 lớp |
| Trả lời câu hỏi | "Chỉ cần đúng loài thì model đúng bao nhiêu % trên cả tập test?" | "Trong số ảnh model đoán sai, bao nhiêu % là lỗi *nhẹ* (nhầm biến thể cùng loài) chứ không phải lỗi *nặng* (nhầm sang loài khác)?" |

Ba con số này liên hệ bằng đúng một công thức — kiểm chứng khớp chính xác với số
liệu thật của `vit_b_16_in21k` (top1=85.86%, err_same_species=14.1%,
err_same_order=92.3%):

```
acc@404sp = top1 + (1 - top1) x err_same_species%
          = 85.86 + (100 - 85.86) x 0.141 = 85.86 + 1.99 = 87.85%   <- khớp đúng bảng

acc@22ord = top1 + (1 - top1) x err_same_order%
          = 85.86 + (100 - 85.86) x 0.923 = 85.86 + 13.05 = 98.91%  <- khớp đúng bảng
```

Nói cách khác: `top1` là điểm khởi đầu; `err_same_species%`/`err_same_order%`
cho biết trong phần **14.14% còn thiếu**, bao nhiêu phần trăm "cứu được" khi nới
tiêu chí đánh giá xuống mức loài/bộ.

### Tổng hợp: cột nào trả lời câu hỏi nào

| muốn biết | dùng cột | tính trên |
|---|---|---|
| Model đúng bao nhiêu % nói chung | `top1` | toàn bộ ảnh test |
| Đáp án đúng có nằm trong "5 khả năng gần nhất" không | `top5` | toàn bộ ảnh test |
| Model có thiên vị lớp đông ảnh không | so `top1` với `macro-F1`/`bal-acc` | so sánh 2 cách trung bình |
| Lớp cụ thể nào yếu | `per_class.csv`, cột `f1`, `top5_recall` | riêng từng lớp |
| Nếu chỉ cần đúng loài (bỏ qua biến thể) thì sao | `acc@404sp` | toàn bộ ảnh test, đã gộp nhãn |
| Nếu chỉ cần đúng bộ thì sao | `acc@22ord` | toàn bộ ảnh test, đã gộp nhãn |
| Trong các lỗi, bao nhiêu % là lỗi nhẹ | `err_same_species%` / `err_same_order%` | chỉ tập con các ảnh đã sai |

## Sự cố vận hành đã gặp (ghi lại để khỏi mất thời gian lần sau)

### 1. Hết RAM vì quá nhiều DataLoader worker

**Triệu chứng:** GPU chỉ 4% utilization, 1.5 GB VRAM, epoch không bao giờ kết thúc.

**Nguyên nhân:** mỗi worker process import torch nên chiếm ~765 MB RSS. Ban đầu
đặt `num_workers=8` cho *cả* train loader và test loader với
`persistent_workers=True` → **16 process sống song song ≈ 12 GB**, cộng main
process ~6 GB. Trên máy 32 GB, `FreePhysicalMemory` tụt xuống **2.5 GB**, hệ điều
hành paging liên tục và GPU bị bỏ đói.

**Cách sửa** (đã áp dụng trong `nabirds_data.py`):
- train loader: `persistent_workers=True`, `--workers 6`
- val loader (chạy mỗi epoch): `num_workers = workers // 2`, **`persistent_workers=True`**
- test loader (chạy đúng 1 lần ở cuối): `persistent_workers=False`

**Đính chính quan trọng:** ban đầu tôi tắt `persistent_workers` cho *cả* val lẫn
test để cứu RAM. Đó là sửa quá tay — xem sự cố #3 bên dưới.

Sau khi sửa, RAM trống giữ ở mức ~13 GB. Kiểm tra nhanh bằng PowerShell:

```powershell
$os = Get-CimInstance Win32_OperatingSystem
"FreeMB={0}" -f [int]($os.FreePhysicalMemory/1KB)
```

Theo bảng benchmark ở trên, `workers=6` vẫn cho ~900 img/s — vượt tốc độ GPU của
mọi model trong sweep, nên không mất gì về throughput.

### 2. `tee` che mất tiến độ

**Triệu chứng:** log chỉ có 3 dòng header, tưởng như process bị treo, trong khi
`nvidia-smi` báo GPU 85%.

**Nguyên nhân:** `python ... | tee file` — `tee` block-buffer khi ghi ra file
(không phải tty), nên output chỉ hiện khi buffer đầy hoặc process kết thúc.
`PYTHONUNBUFFERED=1` và `flush=True` **không** giải quyết được vì nút thắt nằm ở
`tee`, không phải ở python.

**Cách sửa:** bỏ `tee`, redirect thẳng `> "$log" 2>&1`.

**Bài học kèm theo:** đừng dùng `nvidia-smi` một lần để kết luận "GPU đang rảnh".
Giá trị `utilization.gpu` là mẫu tức thời; đọc đúng lúc chuyển giữa train và eval
sẽ thấy 4% dù model đang chạy bình thường. Muốn biết thật thì đo delta CPU-time
của process hoặc xem log tiến độ theo step.

### Chi phí khởi động của mỗi epoch đầu tiên

`train.py` in throughput tích luỹ theo step. Ở `cnn_scratch` quan sát được:

```
[1] step  31/186   63 img/s
[1] step  93/186  145 img/s
[1] step 186/186  232 img/s     -> epoch 1: 103s
```

Throughput tích luỹ tăng dần vì ~30 step đầu tốn khoảng 60s cho cudnn benchmark
autotune + spawn worker. Tốc độ biên ở cuối epoch là **~750 img/s**, nên các
epoch sau chỉ mất ~40s. Đừng hoảng khi epoch 1 chậm gấp 2-3 lần các epoch sau.

### 3. Val loader spawn lại worker mỗi epoch — mất một nửa thời gian

**Triệu chứng:** chu kỳ 1 epoch mất 60s trong khi log báo `train 27s`. GPU rảnh
~50% thời gian. Đo bằng `nvidia-smi` mỗi 2s: 60% số mẫu có GPU < 20%, có đoạn 0%
suốt 30 giây liên tiếp.

**Nguyên nhân:** ở sự cố #1 tôi tắt `persistent_workers` cho val loader để tiết
kiệm RAM. Nhưng val chạy **mỗi epoch**, nên mỗi epoch 3 worker bị giết rồi tạo
lại, và **mỗi worker phải import torch lại từ đầu**.

Đo trực tiếp trên tập val 3,510 ảnh, duyệt hết 3 lần:

| cấu hình | lần 1 | lần 2–3 (TB) |
|---|---|---|
| `persistent=0, workers=3` | 21.6s | **31.3s** |
| `persistent=0, workers=6` | 53.6s | **46.3s** |
| `persistent=1, workers=3` | 41.1s | **7.2s** |
| `persistent=1, workers=6` | 56.7s | **3.4s** |

Chênh ~13x. Tức tôi đã đổi 6 GB RAM lấy 30s mỗi epoch mà không đo lại.

**Cách sửa:** phân biệt val (chạy mỗi epoch → giữ worker sống) với test (chạy
đúng 1 lần ở cuối → không cần). Kết quả sau khi sửa:

```
ep 01/60  ... | train 62s + eval 23s     <- epoch đầu: cudnn autotune + spawn
ep 02/60  ... | train 25s + eval  5s
ep 03/60  ... | train 25s + eval  4s
```

Chu kỳ epoch từ 60s xuống 29s. Tiết kiệm ~30s × tổng số epoch của 6 model ≈ **1.7 giờ**.

**Bài học:** mỗi lần tối ưu một tài nguyên (RAM) phải đo lại tài nguyên kia
(thời gian). `train.py` giờ in `train Xs + eval Ys` để chi phí này luôn nhìn thấy được.

### 4. `TaskStop` không giết bash con — 4 sweep chạy song song

**Triệu chứng:** GPU/RAM/CPU dao động dữ dội và vô lý, log báo `exit 127`, RAM
trống tụt còn 987 MB, có 34 process python trong khi thiết kế chỉ cần 10.

**Nguyên nhân:** mỗi lần dừng sweep để sửa config, việc dừng task chỉ giết
wrapper. Process `bash src/run_all.sh` vẫn sống, chạy tiếp sang model kế tiếp và
tự spawn DataLoader worker mới. Sau 4 lần restart có **4 sweep chạy song song**
tranh nhau GPU và RAM. Mọi số đo tốc độ trong giai đoạn đó đều vô nghĩa.

**Cách kiểm tra:** liệt kê process python theo thời điểm khởi động — process của
sweep hiện tại có `StartTime` cùng mốc, process mồ côi thì lệch hẳn về trước.

```powershell
Get-Process python | Select-Object Id,
  @{n='WS_MB';e={[int]($_.WS/1MB)}},
  @{n='Start';e={$_.StartTime.ToString('HH:mm:ss')}} | Sort-Object Start
```

**Cách dọn triệt để trước khi chạy lại:**

```powershell
Get-CimInstance Win32_Process -Filter "Name='bash.exe'" |
  Where-Object { $_.CommandLine -like '*run_all*' } | Stop-Process -Force
Get-Process python | Stop-Process -Force
```

`run_all.sh` giờ tự kiểm tra và từ chối chạy nếu phát hiện sweep khác đang sống.

### Có nên làm chương trình cache annotation riêng không? — Không

Câu hỏi hợp lý: mỗi run có tốn phần cứng để chuẩn bị data lại từ đầu không? Đo:

| việc | chi phí |
|---|---|
| Đọc cả 6 file annotation | 0.43s |
| Khởi tạo cả 3 dataset (train/val/test) cho 1 run | 0.68s |
| Pickle dataset gửi sang worker | 2.2 MB / 0.01s |
| **Val loader spawn worker mỗi epoch** | **31–46s × mỗi epoch** |

Phần chuẩn bị data nặng thật sự — resize 48,562 ảnh — **đã** là một chương trình
riêng chạy một lần (`src/prepare_images.py`) và dùng chung cho mọi model. Việc
cache thêm manifest annotation chỉ tiết kiệm ~0.7s mỗi run, đổi lại thêm một lớp
phức tạp và một nguồn lỗi đồng bộ mới — không đáng làm.

## Thiết kế tập validation cho bài 555 lớp

NABirds **không** cung cấp tập val, chỉ có train/test. Nhưng nếu chọn checkpoint
tốt nhất bằng chính tập test rồi báo cáo test đó thì con số sẽ lạc quan có hệ
thống — đây là lỗi phương pháp dễ bị phản biện nhất. Nên phải tự tách val từ
train. Với **555 lớp và trung bình chỉ 43 ảnh train/lớp**, việc tách này phải rất
cẩn thận.

### Chọn `val_frac` — đo thực tế

| `val_frac` | n_train | n_val | val/lớp | lớp có 0 ảnh val | lớp ≤2 ảnh val | SE của val-top1 |
|---|---|---|---|---|---|---|
| 0.05 | 22,723 | 1,206 | 2.2 | 9 | 337 | ±1.25% |
| 0.10 | 21,521 | 2,408 | 4.3 | 2 | 79 | ±0.88% |
| **0.15 (chọn)** | 20,332 | 3,597 | 6.5 | **0** | 23 | **±0.72%** |
| 0.20 | 19,149 | 4,780 | 8.6 | 0 | 16 | ±0.63% |

`SE` là sai số chuẩn của val top-1, tính theo `sqrt(p(1-p)/n)` với p≈0.75.

Chọn **0.15**: là mức nhỏ nhất phủ được toàn bộ 555 lớp. Không lên 0.20 vì đổi
0.09% sai số lấy thêm 5% dữ liệu train là không đáng với bài fine-grained.

### Ưu tiên dữ liệu cho lớp hiếm: `--val-min-class 20`

Phân bố ảnh train/lớp: **min 4, p5 17, median 44, max 60**. Các lớp đuôi dài
không được phép mất ảnh nào cho val. Quy tắc: **lớp có <20 ảnh train thì không
góp ảnh cho val**.

| `min_class_size` | n_train | n_val | số lớp có val | SE val-top1 |
|---|---|---|---|---|
| 0 (lấy cả lớp hiếm) | 20,332 | 3,597 | 555 | ±0.72% |
| **20 (chọn)** | **20,419** | 3,510 | 516 | **±0.73%** |
| 30 | 20,688 | 3,241 | 442 | ±0.76% |

Kết quả với ngưỡng 20:
- **39 lớp** (4–19 ảnh train, tổng 546 ảnh = 2.3% tập train) giữ **100%** ảnh cho train
- Verify: `min ảnh train/lớp sau khi tách = 4`, đúng bằng trước khi tách → không
  lớp hiếm nào bị lấy mất ảnh
- Giá phải trả: SE chỉ tăng từ ±0.72% lên ±0.73%

### Calibrate `--min-delta` theo nhiễu của val

Sai số chuẩn của val top-1 là **±0.73%**. Nếu đặt `min_delta` nhỏ hơn mức này
thì early stopping sẽ phản ứng với **nhiễu lấy mẫu** chứ không phải tiến bộ thật.
Nên `--min-delta 0.005` (0.5%), `--patience 8` (12 cho `cnn_scratch` vì model
from-scratch hội tụ chậm và không đều hơn).

### Vì sao val-top1 cao hơn test-top1

Trong log sẽ thấy `val-top1` cao hơn `test-top1` vài điểm (ví dụ 64.4% vs 58.8%
ở smoke test 2 epoch). Đây là chuyện bình thường: val lấy ngẫu nhiên từ cùng
phân phối với train, còn test là split riêng do dataset định nghĩa. Khi theo dõi
training, cái cần nhìn là val có **tăng đều** hay không, không phải giá trị tuyệt
đối của nó. Mọi con số trong `results/` đều tính trên **tập test gốc**, không bị
ảnh hưởng.

### Early stopping có thực sự hữu ích không?

Cần nói thẳng: với cosine annealing về 0 trên một ngân sách epoch cố định,
accuracy thường tăng đến tận epoch cuối, nên early stopping **hiếm khi kích
hoạt**. Nó là **lưới an toàn** chống phân kỳ hoặc overfit chứ không phải công cụ
tiết kiệm thời gian. Đừng kỳ vọng nó cắt ngắn sweep.

## Theo dõi loss trong lúc train

`train.py` in tiến độ 6 lần mỗi epoch:

```
[2] step  112/336  loss(avg) 3.519  loss(batch) 3.200  train-acc 35.52%  lr 3.29e-02  661 img/s  eta 22s
```

- `loss(avg)` — loss trung bình cộng dồn từ đầu epoch, mượt, dùng để xem xu hướng
- `loss(batch)` — loss của batch hiện tại, nhiễu, dùng để phát hiện phân kỳ (NaN/bùng nổ)
- `train-acc` — accuracy cộng dồn trên tập train, so với `val-top1` để thấy overfit
- `lr` — learning rate hiện tại, xác nhận warmup và cosine chạy đúng

Dòng tổng kết mỗi epoch:

```
ep 02/30  train-loss 2.986  train-acc 48.70  |  val-top1 64.41  val-top5 88.41  |  29s *
```

`*` = checkpoint tốt nhất được lưu. `(no-improve k/patience)` = đếm epoch không
cải thiện. `runs/<run>/history.json` lưu lại đầy đủ để vẽ learning curve.

**Dấu hiệu model học thật** (quan sát được ở resnet50, 2 epoch đầu):
`train-loss 6.31 → 2.99`, `train-acc 0.5% → 48.7%`, `val-top1 22.2% → 64.4%`.

## Vận hành: chạy, theo dõi, dừng

```bash
bash src/stop_all.sh      # luôn dọn sạch trước
bash src/run_full.sh      # chạy toàn bộ 8 run (6 @224/299 + 2 @448)
bash src/status.sh        # xem trạng thái bất cứ lúc nào
bash src/preflight.sh     # kiểm tra nhanh 6 model chạy được (~10 phút)
```

| Script | Vai trò |
|---|---|
| `src/run_full.sh` | Điểm vào duy nhất: pass 224/299 → pass 448 → bảng tổng hợp |
| `src/run_all.sh` | Pass 1: 6 model ở độ phân giải gốc của checkpoint |
| `src/run_448.sh` | Pass 2: `cnn_scratch` + `resnet50` ở 448 (dùng cache `images_r512`) |
| `src/preflight.sh` | 8 step train + eval đầy đủ + xuất metrics cho từng model |
| `src/status.sh` | Lịch chạy, epoch/step gần nhất, bảng kết quả, RAM/GPU, cảnh báo chạy chồng |
| `src/stop_all.sh` | Dừng sạch: giết bash sweep theo PID file + mọi worker python |
| `src/sweep_lock.sh` | Khoá chống chạy chồng (`source`, mỗi script một PID file riêng) |

### Vì sao khoá phải dùng PID file

Cách so khớp command line qua WMI **đã từng giết nhầm chính shell đang gọi
script**, vì command line của shell đó cũng chứa chuỗi `run_all`. Ngoài ra
`ps -W` của Git Bash không hiện command line nên guard dựa trên nó không bao giờ
bắt được sweep trùng — đó là lý do có lúc **7 sweep chạy song song**.

Mỗi script giữ một PID file riêng (`runs/.sweep_all.pid`, `.sweep_448.pid`,
`.sweep_full.pid`). Không dùng chung một file vì `run_full.sh` gọi `run_all.sh`,
cả hai cùng sống, một file duy nhất sẽ khiến script con tưởng là có sweep khác.

`stop_all.sh` vẫn phải giết **toàn bộ process python** riêng: trên Windows,
DataLoader worker không chết theo bash cha. Chính chúng mới là thứ ăn hết RAM.

## Cấu hình chạy đã chốt

| Pass | Model | Epoch | Batch | Patience | Ghi chú |
|---|---|---|---|---|---|
| 224 | `cnn_scratch` | 60 | 128 | 12 | from scratch, cần nhiều epoch hơn |
| 224 | `resnet50` | 30 | 64 | 8 | |
| 224 | `resnet101` | 30 | 48 | 8 | |
| 299 | `inception_v3` | 30 | 96 | 8 | bs48 chậm hơn hẳn; batch ×2 → LR ×2 |
| 224 | `vit_b_16` | 25 | 64 | 8 | |
| 224 | `vit_b_16_in21k` | 25 | 64 | 8 | |
| 448 | `cnn_scratch` | 60 | 48 | 12 | cache `images_r512` |
| 448 | `resnet50` | 30 | 32 | 8 | cache `images_r512` |
| 224 | `convnext_tiny` | 25 | 64 | 8 | **AdamW** — xem "Pass 3" cuối file |
| 224 | `convnext_tiny_in22k` | 25 | 64 | 8 | **AdamW** |
| 224 | `efficientnetv2_s` | 25 | 48 | 8 | **AdamW** |
| 224 | `resnet50` (AdamW) | 30 | 64 | 8 | **AdamW** — cầu nối bóc confound optimizer |

Chung: `--workers 12 --val-frac 0.15 --min-delta 0.005`, SGD + cosine,
label smoothing 0.1, bf16 autocast, `channels_last`, freeze backbone 1 epoch đầu.

## Nghẽn ở đâu — đo trên RTX 5070

Trần GPU (dữ liệu tổng hợp, đo lặp 6 lần) so với trần DataLoader:

| model | trần GPU | nghẽn thật sự |
|---|---|---|
| `cnn_scratch` | **1887 img/s** (±2%) | **data** |
| `resnet50` bs64 | 800 img/s (±0%) | GPU |
| `resnet50` bs128 | 637 img/s (±13%) | GPU, và *chậm hơn* bs64 |
| `resnet101` | ~400 img/s | GPU |
| `inception_v3` | 166 (bs48) → 233 (bs96) | GPU |
| `vit_b_16_in21k` | ~330 img/s | GPU |

| DataLoader | throughput |
|---|---|
| 6 worker | 954 img/s |
| 10 worker | 1142 img/s |
| 12 worker | **1059 img/s (đo thực tế trong vòng train)** |
| 14 worker | 1440 img/s |
| 18 worker | 1781 img/s |

**Kết quả sau khi nâng lên 12 worker:** `cnn_scratch` từ 785 → **1059 img/s**,
train/epoch từ 26s → **19s**, chu kỳ epoch từ 31s → **23s** (nhanh hơn 26%).

### Ba điều dễ kết luận sai

1. **Batch size lớn hơn KHÔNG nhanh hơn.** `resnet50` ở bs128 cho 637 img/s so
   với 800 ở bs64, lại còn dao động 13%. VRAM nhỏ không phải là lãng phí. Ngoại
   lệ duy nhất là `inception_v3`.
2. **`nvidia-smi utilization.gpu` gần như vô dụng.** Đo được 2% trong khi power
   draw là 149.8 W và throughput 1059 img/s (idle chỉ 26 W). Nó là mẫu tức thời
   trên cửa sổ rất ngắn. Nhìn `img/s` trong log hoặc `power.draw` thay vì nó.
3. **Đo benchmark thiếu warmup cho kết quả sai hoàn toàn.** Với 4 step warmup,
   chi phí cudnn autotune bị dồn hết vào cấu hình đo đầu tiên trong mỗi nhóm,
   tạo ra "cải thiện" giả +35% đến +188% khi tăng batch. Cần ≥15 step warmup và
   đo lặp nhiều lần mới thấy sự thật là phần lớn phẳng.

### `torch.compile` không dùng được trên Windows

Thiếu Triton (`TritonMissing`). `triton-windows` là gói bên thứ ba còn thử
nghiệm. Đây là hạn chế thật của nền tảng — trên Linux đòn bẩy này sẽ có.

## Khởi tạo head thống nhất

Với 555 lớp, loss lúc khởi tạo phải xấp xỉ `ln(555) = 6.32`. Pre-flight cho thấy
`vit_b_16_in21k` bắt đầu ở **8.5** trong khi 5 model kia đều ~6.4 — head mới của
timm khởi tạo trọng số quá lớn, gradient lớn ở vài step đầu dễ phá trọng số
pretrain của backbone. `models_zoo._init_head` giờ khởi tạo mọi head bằng
`trunc_normal(std=0.01)` + bias 0:

| model | loss lúc khởi tạo |
|---|---|
| `cnn_scratch` | 6.32 |
| `resnet50` | 6.19 |
| `resnet101` | 6.40 |
| `vit_b_16` | 6.36 |
| `vit_b_16_in21k` | 6.98 (từ 8.5) |
| `inception_v3` | 6.21 |

Phần dư của `vit_b_16_in21k` đến từ độ lớn đặc trưng của backbone, không phải
head. Quan trọng là mọi model xuất phát từ cùng một điểm nên so sánh mới công bằng.

## Đính chính dự đoán: CNN from-scratch

Tôi dự đoán `cnn_scratch` đạt **20–35%** top-1. Kết quả thực tế là **60.72%** —
sai gần gấp đôi.

Nguyên nhân sai: tôi ước lượng theo các kết quả from-scratch công bố trên
**CUB-200-2011** (5,994 ảnh train). NABirds có **20,419 ảnh train** sau khi tách
val, gấp 3.4 lần, và chính lượng dữ liệu đó mới quyết định model from-scratch
học được đến đâu. Bài học: đừng chuyển thẳng con số baseline giữa hai dataset
chỉ vì cùng miền ảnh.

### Kết quả `cnn_scratch` @224 (5.0M params, 60 epoch, không pretrain)

| chỉ số | giá trị |
|---|---|
| top-1 | **60.72%** |
| top-5 | 83.85% |
| balanced accuracy | 57.44% |
| macro P / R / F1 | 59.08 / 57.44 / **57.22** |
| weighted F1 | 60.21 |
| accuracy @404 species | 62.49% |
| accuracy @22 order | 87.65% |
| lỗi nằm trong cùng loài | 4.52% |
| lỗi nằm trong cùng order | 68.55% |

**Overfit rõ:** train-acc 83.51% vs test 60.72% — chênh 22.8 điểm.

**Val lạc quan hơn test 7.4 điểm** (68.09% vs 60.72%), đúng như đã cảnh báo ở
mục thiết kế val: val lấy ngẫu nhiên từ cùng phân phối train, còn test là split
riêng của dataset. Không ảnh hưởng con số báo cáo (test chỉ chạy một lần ở cuối)
nhưng cần nhớ khi đọc log lúc đang train.

**Early stopping không kích hoạt** (dừng ở 7/12 epoch không cải thiện). Đúng như
dự đoán: với cosine annealing về 0, accuracy cải thiện tới tận epoch cuối, nên
early stopping là lưới an toàn chứ không phải công cụ tiết kiệm thời gian.

**68.55% lỗi nằm trong cùng order** — model đã nắm được cấu trúc thô (order-level
87.65%) và phần lớn sai sót là nhầm giữa các loài gần nhau, đúng đặc trưng của
bài fine-grained.

---

# KẾT QUẢ CUỐI — NABirds 555 lớp

8 run, tổng ~5 giờ trên RTX 5070 12GB. Giao thức: split gốc (20,419 train sau
khi tách val / 24,633 test), **không dùng bbox lúc test**, test chỉ chạy một lần
với checkpoint tốt nhất theo val.

## Bảng tổng hợp

| run | params | pretrain | input | top-1 | top-5 | bal-acc | macro-F1 | @404 sp | @22 ord | phút |
|---|---|---|---|---|---|---|---|---|---|---|
| **`vit_b_16_in21k`** | 86.2M | **ImageNet-21k** | 224 | **85.86%** | 97.53% | 83.67% | **83.91%** | 87.85% | 98.91% | 28 |
| `resnet50` | 24.6M | ImageNet-1k | **448** | 83.18% | 96.31% | 80.61% | 80.74% | 85.06% | 97.67% | 58 |
| `inception_v3` | 25.9M | ImageNet-1k | 299 | 80.93% | 95.25% | 78.78% | 78.80% | 82.52% | 97.15% | 24 |
| `resnet101` | 43.6M | ImageNet-1k | 224 | 79.56% | 94.73% | 76.91% | 76.97% | 81.55% | 97.04% | 24 |
| `vit_b_16` | 86.2M | ImageNet-1k | 224 | 79.19% | 95.25% | 76.72% | 76.83% | 80.90% | 97.10% | 33 |
| `resnet50` | 24.6M | ImageNet-1k | 224 | 78.61% | 94.40% | 75.81% | 75.86% | 80.50% | 96.70% | 17 |
| `cnn_scratch` | 5.0M | không | **448** | 63.74% | 85.55% | 60.22% | 60.08% | 65.59% | 89.16% | 89 |
| `cnn_scratch` | 5.0M | không | 224 | 60.72% | 83.85% | 57.44% | 57.22% | 62.49% | 87.65% | 26 |

`micro-P = micro-R = micro-F1 = top-1` (single-label multiclass), nên không lặp
lại trong bảng. Đầy đủ ở `results/comparison.csv`.

## Bốn đòn bẩy, xếp theo hiệu quả thực đo

| đòn bẩy | thí nghiệm đối chứng | điểm | chi phí |
|---|---|---|---|
| **Có pretrain vs không** | `cnn_scratch` → `resnet50`, cùng 224 | **+17.89** | miễn phí |
| **Nguồn pretrain IN1k → IN21k** | `vit_b_16` → `vit_b_16_in21k`, mọi thứ khác giống hệt | **+6.67** | miễn phí |
| **Độ phân giải 224 → 448** | `resnet50` cùng weights, cùng recipe | **+4.57** | 3.4x thời gian |
| **Độ sâu R50 → R101** | cùng pretrain, cùng 224 | **+0.95** | 1.8x tham số |

Đúng thứ tự dự đoán từ bảng SOTA lúc nghiên cứu: **pretrain > độ phân giải > kiến trúc**.

### Thí nghiệm sạch nhất: nguồn pretrain

`vit_b_16` và `vit_b_16_in21k` **cùng kiến trúc ViT-B/16, cùng 86.2M tham số,
cùng 224px, cùng recipe**. Biến duy nhất là nguồn pretrain:

| | ImageNet-1k | ImageNet-21k | chênh |
|---|---|---|---|
| top-1 | 79.19% | 85.86% | **+6.67** |
| macro-F1 | 76.83% | 83.91% | **+7.08** |

Tái lập được luận điểm trung tâm của Cui et al. CVPR'18 (họ đo +5.9 điểm khi đổi
ImageNet → iNaturalist trên chính NABirds).

### Độ phân giải cộng dồn với pretrain

| model | 224 | 448 | chênh |
|---|---|---|---|
| `cnn_scratch` (không pretrain) | 60.72% | 63.74% | +3.02 |
| `resnet50` (pretrain IN1k) | 78.61% | 83.18% | **+4.57** |

Model có pretrain **hưởng lợi nhiều hơn** từ độ phân giải cao. Hai đòn bẩy không
chồng lấn nhau. `resnet50` @448 (24.6M) còn vượt `resnet101` @224 (43.6M)
**3.62 điểm** — cho cùng ngân sách, tăng độ phân giải đáng giá hơn tăng độ sâu.

## Đối chiếu với paper

| | top-1 NABirds |
|---|---|
| Ours: ViT-B/16 IN21k @224, không module thêm | **85.86%** |
| TransFG (AAAI'22): ViT-B/16 IN21k **@448** + Part Selection Module | 90.8% |
| MetaFormer-0: iNat2021 @384 | 91.5% |
| Cui et al. CVPR'18: Inception-v3 ImageNet @560 | 82.0% |
| Cui et al. CVPR'18: Inception-v3 **iNaturalist** @560 | 87.9% |

Khoảng cách ~5 điểm với TransFG chủ yếu là **độ phân giải (224 vs 448)**, khớp
với mức +4.57 đo được khi nâng `resnet50` lên 448. Đáng chú ý: `inception_v3`
@299 của ta đạt 80.93%, gần bằng 82.0% của Cui et al. ở 560px.

## Phân tích lỗi

### Lỗi dồn vào đâu khi model tốt lên

| model | top-1 | lỗi cùng loài | lỗi cùng order |
|---|---|---|---|
| `cnn_scratch` @224 | 60.72% | 4.5% | 68.6% |
| `resnet50` @224 | 78.61% | 8.8% | 84.6% |
| `resnet101` @224 | 79.56% | 9.7% | 85.5% |
| `vit_b_16_in21k` | 85.86% | **14.1%** | **92.3%** |

Model càng mạnh, lỗi càng dồn vào phân biệt **biến thể giới tính/tuổi trong cùng
một loài** — đúng loại khó nhất, và khớp với phát hiện ban đầu rằng 288/555 lớp
là biến thể chứ không phải loài riêng.

### 10 cặp nhầm nhiều nhất (`vit_b_16_in21k`)

| lớp thật | bị nhầm thành | số ảnh |
|---|---|---|
| Allen's Hummingbird (Adult Male) | Rufous Hummingbird (Adult Male) | 23 |
| Red-naped Sapsucker | Yellow-bellied Sapsucker | 22 |
| Black-capped Chickadee | Carolina Chickadee | 21 |
| Semipalmated Sandpiper | Western Sandpiper | 20 |
| Cooper's Hawk (Adult) | Sharp-shinned Hawk (Adult) | 19 |
| Greater Scaup (Breeding male) | Lesser Scaup (Breeding male) | 19 |
| Greater Yellowlegs | Lesser Yellowlegs | 19 |
| Great Cormorant (Adult) | Double-crested Cormorant (Adult) | 18 |
| Greater Scaup (Female/Eclipse male) | Lesser Scaup (Female/Eclipse male) | 18 |
| Great-tailed Grackle | Boat-tailed Grackle | 18 |

**100% trong 60 cặp nhầm nhiều nhất là cùng order.** Và đây đều là những cặp mà
người xem chim có kinh nghiệm cũng thường nhầm: Allen's/Rufous Hummingbird,
Cooper's/Sharp-shinned Hawk, Greater/Lesser Scaup, Greater/Lesser Yellowlegs,
Black-capped/Carolina Chickadee. Model không sai lung tung — nó sai đúng chỗ khó.

### F1 theo số ảnh test mỗi lớp

| số ảnh/lớp | số lớp | F1 TB | recall TB | precision TB |
|---|---|---|---|---|
| ≤20 | 31 | 76.3 | 72.3 | 84.6 |
| 21–30 | 77 | **70.7** | 68.2 | 76.3 |
| 31–40 | 111 | 83.3 | 82.9 | 84.9 |
| 41–50 | 105 | 85.8 | 85.6 | 86.9 |
| 51–60 | 231 | **88.8** | 89.9 | 88.4 |

Chênh lệch **18 điểm F1** giữa nhóm lớp hiếm và nhóm lớp nhiều ảnh. Nhóm ≤20 ảnh
lại **cao hơn** nhóm 21–30 — vì các lớp cực hiếm thường là loài có ngoại hình
riêng biệt (Parrots, Storks) chứ không phải loài khó phân biệt.

### Order dễ nhất và khó nhất (`vit_b_16_in21k`)

| order | số lớp | accuracy | macro-F1 |
|---|---|---|---|
| Swifts and Hummingbirds | 21 | **65.35%** | 63.35% |
| Frigatebirds, Boobies, Cormorants… | 10 | 73.46% | 67.94% |
| Skuas and Alcids | 4 | 74.73% | 70.72% |
| Hawks, Kites, Eagles, and Allies | 29 | 76.43% | 75.60% |
| Gulls, Terns, and Allies | 26 | 77.55% | 73.96% |
| … | | | |
| Grouse, Quail, and Allies | 10 | 96.78% | 94.27% |
| Storks | 1 | 98.21% | 96.49% |
| Kingfishers and Allies | 1 | 98.33% | 97.52% |
| Parrots | 1 | **100.00%** | 100.00% |

Hummingbirds khó nhất (65.35%) vì các con cái/non của nhiều loài gần như không
phân biệt được. Order chỉ có 1 lớp thì gần như luôn đúng — đó là bài toán dễ hơn
hẳn, cần lưu ý khi diễn giải.

## Đính chính dự đoán ban đầu

| model | tôi dự đoán | thực tế | sai lệch |
|---|---|---|---|
| `cnn_scratch` @224 | 20–35% | **60.72%** | **sai nặng** |
| `resnet50` @224 | 76–80% | 78.61% | đúng |
| `resnet50` @448 | 82–85% | 83.18% | đúng |
| `inception_v3` | — | 80.93% | — |
| `vit_b_16_in21k` @224 | 88–89% (ước cho 384) | 85.86% @224 | hơi lạc quan |

Sai lầm ở `cnn_scratch`: tôi lấy con số from-scratch công bố trên **CUB-200-2011**
(5,994 ảnh train) áp cho NABirds (20,419 ảnh train, gấp 3.4 lần). Lượng dữ liệu
mới là thứ quyết định model from-scratch học được đến đâu — đừng chuyển thẳng
baseline giữa hai dataset chỉ vì cùng miền ảnh.

## File kết quả

Mỗi run có `results/<run_name>/`:

| file | nội dung |
|---|---|
| `summary.json` | toàn bộ chỉ số tổng hợp |
| `per_class.csv` | **555 dòng**: support, TP/FP/FN, precision, recall, F1, accuracy, top-5 recall, lớp bị nhầm sang nhiều nhất |
| `per_species.csv` | 404 dòng gộp theo loài |
| `per_order.csv` | 22 dòng gộp theo bộ |
| `confusions.csv` | 60 cặp nhầm nhiều nhất, kèm cờ cùng loài / cùng order |

Bảng so sánh chung: `results/comparison.csv` và `results/comparison.md`.

```bash
python src/report.py                      # bảng so sánh mọi run
python src/report.py --run <ten_run>      # chi tiết 1 run
```


---

# Pass 3 — CNN hiện đại (`src/run_cnn2.sh`)

Pass 1 và 2 chỉ có CNN thế hệ 2015–2016 (ResNet, Inception-v3), đều pretrain
ImageNet-1k. Hai câu hỏi vì thế còn bỏ ngỏ:

1. **Trục kiến trúc.** Thiết kế CNN 2022 (ConvNeXt) hơn ResNet-50 bao nhiêu, khi
   giữ nguyên nguồn pretrain và độ phân giải?
2. **Trục pretrain trên CNN.** Chênh +6.67 điểm khi đổi IN1k → IN21k mới chỉ đo
   được trên ViT. Nó có lặp lại trên CNN không, hay là đặc thù của ViT?

Bốn run được chọn để mỗi run trả lời đúng một câu:

| run | pretrain | params | so với ai | trả lời câu gì |
|---|---|---|---|---|
| `convnext_tiny` | IN1k | 28.2M | `resnet50` @224 IN1k | trục kiến trúc |
| `convnext_tiny_in22k` | IN22k | 28.2M | dòng trên | trục pretrain trên CNN |
| `efficientnetv2_s` | IN21k | 20.9M | hai dòng trên | trục pretrain có đặc thù ConvNeXt không |
| `resnet50` + AdamW | IN1k | 24.6M | `convnext_tiny` | cầu nối bóc confound optimizer |

## Bẫy: recipe SGD của repo KHÔNG chuyển sang ConvNeXt được

Toàn bộ pass 1 và 2 chạy SGD momentum. Trước khi chạy pass 3 tôi probe 4 epoch,
hai nhánh giống hệt nhau về mọi thứ trừ optimizer (`runs/logs/_probe_cnx.log`):

| model | optimizer | lr backbone / head | val top-1 @ep4 | test top-1 @4ep |
|---|---|---|---|---|
| `convnext_tiny` | **AdamW** | 1e-4 / 1e-3 (wd 0.05) | **81.91** | 75.91 |
| `convnext_tiny` | SGD | 1e-3 / 1e-2 (wd 1e-4) | 32.54 | 29.25 |

**Chênh 49 điểm.** Nếu cứ chạy ConvNeXt bằng SGD như phần còn lại của repo thì
kết luận rút ra sẽ là "ConvNeXt tệ hơn ResNet-50" — sai hoàn toàn, và sai theo
kiểu rất khó phát hiện vì đường loss vẫn giảm đều, không có dấu hiệu nào của lỗi.

Điều cần cẩn thận khi diễn giải: **đây không phải chuyện LayerNorm.** Đối chiếu
val top-1 tại cùng epoch 4 của các run đã có:

| model | optimizer | lr backbone / head | val top-1 @ep4 |
|---|---|---|---|
| `vit_b_16_in21k` (LayerNorm) | SGD | 1e-3 / 1e-2 | **86.84** |
| `resnet50` (BatchNorm) | SGD | 5e-3 / 5e-2 | 76.70 |
| `convnext_tiny` (LayerNorm) | SGD | 1e-3 / 1e-2 | **32.54** |

ViT cũng dùng LayerNorm và chạy tốt với **đúng mức lr SGD đó**. Kết luận đúng là
hẹp hơn: *mức lr không chuyển được từ ViT sang ConvNeXt*, chứ không phải "SGD
không dùng được cho ConvNeXt". Không dò thêm lr cho SGD vì AdamW là recipe gốc
của cả ConvNeXt lẫn EfficientNetV2 và đã cho kết quả tốt ngay lần đầu.

**Hệ quả về phương pháp:** trục kiến trúc `resnet50`(SGD) vs `convnext_tiny`(AdamW)
có confound optimizer. Vì vậy pass 3 chạy thêm `resnet50_224_adamw` với **đúng bộ
cờ AdamW của ConvNeXt** — so AdamW-với-AdamW thì confound biến mất.


## Kết quả pass 3

| run | pretrain | params | input | top-1 | top-5 | macro-F1 | @404 sp | phút |
|---|---|---|---|---|---|---|---|---|
| `convnext_tiny_in22k` | **IN22k** | 28.2M | 224 | **85.55** | 97.19 | 83.23 | 87.40 | **17.5** |
| `efficientnetv2_s` | IN21k | 20.9M | 224 | 83.47 | 96.11 | 81.01 | 85.43 | 35.3 |
| `convnext_tiny` | IN1k | 28.2M | 224 | 81.49 | 95.23 | 78.86 | 83.30 | 17.9 |
| `resnet50` + AdamW | IN1k | 24.6M | 224 | 79.37 | 94.31 | 76.52 | 81.26 | 20.3 |
| *(tham chiếu)* `resnet50` + SGD | IN1k | 24.6M | 224 | 78.61 | 94.40 | 75.86 | 80.50 | 16.9 |

### 1. Cầu nối optimizer làm đúng việc của nó

`resnet50` với **đúng bộ cờ AdamW của ConvNeXt**: 79.37 vs 78.61 của bản SGD →
**+0.76 điểm là do optimizer**. Nhờ đó tách được trục kiến trúc cho sạch:

| so sánh | chênh thô | quy về đâu |
|---|---|---|
| `resnet50`(SGD) → `convnext_tiny`(AdamW) | +2.88 | lẫn hai thứ |
| `resnet50`(**AdamW**) → `convnext_tiny`(**AdamW**) | **+2.12** | **kiến trúc** |
| `resnet50`(SGD) → `resnet50`(AdamW) | **+0.76** | **optimizer** |

Không có run cầu nối thì +0.76 điểm này bị quy oan cho kiến trúc.

### 2. Trục pretrain lặp lại được trên CNN, nhưng nhỏ hơn trên ViT

| kiến trúc | IN1k | IN21k/22k | chênh |
|---|---|---|---|
| ViT-B/16 (86.2M) | 79.19 | 85.86 | **+6.67** |
| ConvNeXt-T (28.2M) | 81.49 | 85.55 | **+4.06** |

`efficientnetv2_s` (IN21k, họ CNN khác) đạt 83.47 — vượt mọi CNN IN1k @224 trong
repo, xác nhận hướng của trục. **Không** đọc 83.47 < 85.55 thành "ConvNeXt hơn
EfficientNetV2 về kiến trúc": cfg gốc của `tf_efficientnetv2_s` là **300px**, ở
đây bị ép về 224 để nằm chung trục độ phân giải, nên đó là cận **dưới**.

### 3. ConvNeXt-T IN22k ngang ViT-B/16 IN21k với 1/3 tham số

| | params | phút | top-1 | macro-F1 | @404 sp |
|---|---|---|---|---|---|
| `vit_b_16_in21k` | 86.2M | 28.3 | 85.86 | 83.91 | 87.85 |
| `convnext_tiny_in22k` | **28.2M** | **17.5** | 85.55 | 83.23 | 87.40 |
| chênh | −67% | −38% | **−0.31** | −0.68 | −0.45 |

Chênh −0.31 nhỏ hơn sai số chuẩn của val (~±0.72), nên đọc đúng là **ngang nhau
về độ chính xác, rẻ hơn hẳn về chi phí**. Nó cũng vượt bản LoRA của ViT (85.06):
muốn tiết kiệm thì **đổi kiến trúc hiệu quả hơn đổi cách fine-tune** — và không
kèm cái giá "chậm hơn 38%" mà LoRA phải trả.

### 4. Bảng đòn bẩy sau khi cập nhật

| đòn bẩy | thí nghiệm đối chứng | điểm |
|---|---|---|
| Có pretrain vs không | `cnn_scratch` → `resnet50`, cùng 224 | **+17.89** |
| Nguồn pretrain IN1k → IN21k (**ViT**) | `vit_b_16` → `vit_b_16_in21k` | **+6.67** |
| Độ phân giải 224 → 448 | `resnet50` cùng weights, cùng recipe | **+4.57** |
| Nguồn pretrain IN1k → IN22k (**CNN**) | `convnext_tiny` → `convnext_tiny_in22k` | **+4.06** |
| Kiến trúc CNN 2016 → 2022 | `resnet50`(AdamW) → `convnext_tiny`(AdamW) | **+2.12** |
| Độ sâu R50 → R101 | cùng pretrain, cùng 224, cùng SGD | +0.95 |
| Optimizer SGD → AdamW | `resnet50` cùng mọi thứ khác | +0.76 |

Thứ tự **pretrain > độ phân giải > kiến trúc** vẫn đứng, nhưng phải nói chính xác
hơn sau pass 3: trục pretrain trên CNN (+4.06) và trục độ phân giải (+4.57) là
**xấp xỉ nhau**; chỉ trên ViT thì pretrain mới bỏ xa (+6.67). Kiến trúc (+2.12)
và độ sâu (+0.95) vẫn ở nhóm cuối.

Một quan sát sắc hơn ở đáy bảng: **đòn bẩy optimizer (+0.76) gần bằng đòn bẩy độ
sâu (+0.95)**. `resnet50`+AdamW đạt 79.37, chỉ kém `resnet101`+SGD (79.56) đúng
0.19 điểm — tức gần hết khoảng cách do nhân đôi độ sâu tạo ra thu hồi được **miễn
phí** bằng cách đổi optimizer. Chưa chạy `resnet101`+AdamW nên không kết luận
được chiều ngược lại.

### 5. Ít tham số vẫn không có nghĩa là nhanh hơn

`efficientnetv2_s` **ít tham số nhất** trong pass 3 (20.9M) nhưng **chậm nhất**
(35.3 phút, gấp đôi ConvNeXt-T 28.2M / 17.5 phút). Depthwise conv nghẽn ở băng
thông bộ nhớ chứ không ở FLOPs. Đây là lần thứ hai repo gặp đúng bài học này —
lần đầu là LoRA (3.2% tham số nhưng chậm hơn full fine-tune 38%).

---

# Pass 6 — Augmentation (`src/run_aug.sh`, `src/run_aug2.sh`)

## 0. Vì sao phải tra tài liệu trước khi chạy

Bản kế hoạch đầu tôi tự suy diễn, và tra cứu cho thấy **phần lớn suy diễn đó sai
hướng**. Ba điều tài liệu nói ngược trực giác:

**(a) Trộn ảnh KHÔNG hại cho chim — nếu train đủ dài.** Benchmark OpenMixup trên
CUB-200 (transfer learning, **200 epoch**): CutMix **+2.67** và SAMix **+3.82**
trên ResNeXt-50; chỉ FMix là âm (−0.40 trên ResNet-18). Ghi chú cũ của `docs/02`
("mixup/CutMix với FGVC thường **hại** hơn lợi") **sai về dấu**. Phần đúng là
điều kiện **số epoch** — repo chạy 25–30 epoch, benchmark chạy 200.

**(b) Xoá vùng ngẫu nhiên (Cutout) làm hại FGVC** — tài liệu ghi ResNet-50 mất
~2 điểm trên CUB vì có xác suất xoá trúng vùng phân biệt, mà ở chim vùng đó rất
nhỏ. Repo bật `RandomErasing(p=0.25)` mặc định và chưa ai kiểm.

**(c) Sinh ảnh bằng diffusion là lựa chọn TỆ NHẤT cho chim.** SaSPA (NeurIPS
2024) trên 5 dataset: CompCars +5.7, DTD +2.2, Aircraft +1.7, Cars +1.1, **CUB
+0.7 — thấp nhất**. Tác giả thừa nhận CUB kém vì *"class-fidelity thấp trên
dataset phức tạp"*, tức model sinh ảnh **không giữ được đặc điểm loài** khi các
loài quá giống nhau. Chi phí: 4× RTX 3090. ⇒ **Bỏ hẳn** — lợi nhỏ nhất đúng ở
miền chim, chi phí lớn nhất, và rủi ro sinh ra ảnh SAI loài chính là thứ đang
giết Greater vs Lesser Scaup.

## 1. Đo cái gì — top-1 tổng KHÔNG đủ

113/555 lớp có <30 ảnh train nhưng chỉ chiếm ~10% ảnh test, nên một cải thiện
thật ở đó chìm nghỉm trong top-1. `src/aug_report.py` tách bốn con số:

1. top-1 tổng — để chắc không đánh đổi ngược
2. **macro-F1 trên 113 lớp đuôi** (<30 ảnh train)
3. **F1/recall trên 20 lớp KHÓ** — định nghĩa là 20 lớp có F1 **trung bình thấp
   nhất trên TOÀN BỘ run**, không lấy từ một model nào
4. `err_same_species%`

> **Bẫy phương pháp đã mắc và đã sửa.** Bản đầu tôi lấy danh sách lớp khó từ
> `results/bioclip_224/per_class.csv` rồi so bioclip trên chính nó — **selection
> bias**: chọn lớp VÌ model đó thua, rồi kết luận model đó thua. Chênh lệch giữa
> hai cách rất lớn: recall@hard của bioclip là **1.57** theo tập thiên vị và
> **28.08** theo tập độc lập, tức bản sai phóng đại ~18 lần.
>
> `Northwestern Crow` bị loại thẳng khỏi mọi tập mục tiêu: AOS đã **gộp** nó vào
> `American Crow` năm 2020, nên F1 = 0.00 của nó là vấn đề **nhãn**, không phải
> vấn đề model — không augmentation nào sửa được.

## 2. Kết quả pass 6a — `resnet50` @224, mỗi run đổi ĐÚNG một biến

| cấu hình | top-1 | macro-F1 | F1@113 lớp đuôi | recall@20 lớp khó | kết luận |
|---|---|---|---|---|---|
| baseline (30 ep) | 78.61 | 75.86 | 62.04 | 22.70 | — |
| `--erasing-p 0` | **78.69** | 76.19 | 63.18 ⚠ | 24.41 ⚠ | ⚠ không lặp lại |
| `--rotate 15` | 78.50 | 76.06 | **63.29** | **25.43** | ✅ giữ |
| `--hue 0.10` | **77.92** | 75.26 | **61.91** | 23.40 | ❌ bỏ |
| 100 epoch | **80.25** | 77.64 | **64.30** | 22.37 | ✅ đòn bẩy lớn nhất |
| 100 epoch + CutMix | **80.48** | 77.87 | 63.87 | **24.11** | ✅ nhưng nhỏ |

### 2.1 Top-1 tổng che mất tác dụng thật — nhưng chỉ với xoay

Xoay 15° cho **−0.11** top-1 (trông như có hại) nhưng nâng nhóm đuôi **+1.25** và
nhóm khó **+2.73**. Nếu repo chỉ theo dõi top-1 như trước pass 6 thì nó **đã bị
kết luận là bỏ đi**. Đây là lý do `aug_report.py` phải tồn tại.

### 2.1b Tắt `RandomErasing`: KHÔNG lặp lại được — ghi lại như một kết quả ÂM

Chạy trên model thứ hai cho kết quả **ngược dấu** ở đúng nhóm chỉ tiêu:

| | top-1 | F1@đuôi | recall@khó |
|---|---|---|---|
| `resnet50` −erasing | +0.08 | **+1.14** | **+1.71** |
| `convnext_tiny_in22k` −erasing | +0.13 | **−0.49** | **−0.45** |

Top-1 nhích lên ở cả hai (+0.08 / +0.13) nhưng đều dưới ngưỡng nhiễu; nhóm
đuôi/khó thì một model lên một model xuống. **Kết luận đúng: không có hiệu ứng
nhất quán.** Con số trên `resnet50` nhiều khả năng nằm trong nhiễu của 1 seed.

Bài học phương pháp: tôi đã suýt ghi E1 thành "thắng lợi" sau khi chỉ chạy MỘT
model. Tài liệu dự đoán Cutout hại FGVC ~2 điểm, kết quả đầu tiên khớp hướng, và
điều đó đủ để tạo cảm giác đã xác nhận. Model thứ hai bác bỏ. Lý do vật lý có
thể đơn giản: `RandomErasing(scale=0.02–0.15)` của repo **nhẹ hơn hẳn** Cutout
chuẩn nên không tạo ra hiệu ứng đo được ở mức nhiễu này.

### 2.2 D3 được xác nhận bằng số, và tổn thất dồn đúng chỗ dự đoán

Quyết định **D3** (*"hue jitter ~ 0 vì màu bộ lông CHÍNH LÀ nhãn"*) trước giờ chỉ
là lập luận. Tăng hue lên 0.10 rồi tách F1 theo nhóm loài:

| nhóm lớp | số lớp | F1 (hue 0.02) | F1 (hue 0.10) | chênh |
|---|---|---|---|---|
| **màu là nhãn** — Warbler, Tanager, Oriole, Finch, Bunting, Redstart | 77 | 79.41 | 78.19 | **−1.22** |
| còn lại | 478 | 75.29 | 74.79 | −0.50 |

Tổn thất **gấp 2.4 lần** ở nhóm mà màu là đặc điểm phân biệt (Yellow vs Orange
Warbler). Đúng cơ chế D3 dự đoán, không phải trùng hợp.

`hue 0.10` cũng là phép biến đổi **duy nhất làm hại cả nhóm đuôi** (61.91 vs
62.04). Nên kết luận không phải "augmentation mạnh hơn thì tốt cho lớp hiếm" —
nó phụ thuộc phép biến đổi có **phá tín hiệu nhãn** hay không.

### 2.3 Ngân sách epoch lớn hơn cả kiến trúc

`resnet50` chạy 100 epoch đạt **80.25**, best epoch **73** (bản 30 epoch dừng ở
best@23). Xếp cạnh các đòn bẩy khác đo trên cùng model:

| đòn bẩy | điểm |
|---|---|
| **Ngân sách epoch 30 → 100** | **+1.64** |
| Grouped conv (→ ResNeXt-50, 30 ep) | +1.08 |
| Độ sâu (→ ResNet-101, 30 ep) | +0.95 |

⇒ **Mọi run CNN 30-epoch trong repo đang thiếu train ~1.6 điểm.** Các so sánh
kiến trúc vẫn công bằng nội bộ (đều 30 epoch) nhưng con số tuyệt đối đều thấp
hơn khả năng thật.

Về cơ chế: train lâu giúp **lớp đuôi** (+2.26 F1) nhưng **không giúp 20 lớp khó
nhất** (22.70 → 22.37, hơi giảm). Nhóm khó bị chặn bởi đặc trưng/độ phân giải
chứ không phải bởi tối ưu hoá — chỉ độ phân giải 448 mới nâng được nó (+5.45).

### 2.4 CutMix: dương, nhưng nhỏ hơn nhiều so với CUB

So với baseline **100 epoch riêng** (không phải bản 30 epoch — so như thế là lỗi
đo): **+0.23** top-1, **+1.74** recall lớp khó, **−0.43** F1 lớp đuôi.

Benchmark OpenMixup đo CutMix **+2.67** trên CUB. Ta được +0.23. Hai lý do:
1. Mới 100 epoch chứ chưa phải 200.
2. **NABirds có 23,929 ảnh train, gấp ~4 lần CUB (5,994).** CutMix là bộ điều
   chuẩn — giá trị của nó co lại khi dữ liệu dồi dào hơn. Con số của CUB **không
   chuyển thẳng sang được**.

Ghi chú cũ "mixup thường hại hơn lợi" **sai về dấu**; con số đúng là dương-nhưng-nhỏ.

## 3. CMO — dán chim lớp hiếm lên nền lớp nhiều ảnh (pass 6b)

Park et al., *The Majority Can Help The Minority*, CVPR 2022. Ý tưởng: lớp thiểu
số thiếu **bối cảnh đa dạng**, nên mượn nền của lớp đa số.

**Bản của repo tốt hơn bản gốc ở một điểm:** CMO gốc cắt một **ô ngẫu nhiên** từ
ảnh thiểu số (nên có thể trượt mất con chim), còn NABirds có **bbox thật cho mọi
ảnh** nên dán được đúng con chim.

Cài đặt (`--cmo-p`, `--cmo-tail-max`):
- Pool nguồn = **2,377 ảnh** thuộc 113 lớp có <30 ảnh train, tính trên train đã
  trừ val.
- Dán **SAU** transform, trên tensor. Dán trước thì `RandomResizedCrop` có thể
  cắt mất đúng con chim vừa dán → nhiễu nhãn, đúng điểm yếu SnapMix chỉ ra ở
  CutMix.
- Tỉ lệ cạnh dán ~ U(0.3, 0.7) ⇒ diện tích 9–49%; `lam = 1 − diện tích dán`.
- Loss dùng `reduction='none'` vì **`lam` khác nhau theo TỪNG ẢNH** (diện tích
  dán ngẫu nhiên), không phải theo batch như mixup.

Pass 6b chạy 12 run: quét `cmo-p` ∈ {0.25, 0.5, 0.75}, kiểm tra cộng dồn với
xoay 15°, và **bộ ba** (mặc định → `_noerase` → `_noerase + cmo`) cho ViT-B/16,
ConvNeXt-T IN22k, BioCLIP — bộ ba là bắt buộc, nếu chỉ chạy bản có CMO thì không
tách được đóng góp của CMO khỏi đóng góp của việc tắt erasing.

## 4. Nguồn

- Park et al., *The Majority Can Help The Minority: Context-rich Minority
  Oversampling for Long-tailed Classification*, CVPR 2022, arXiv:2112.00412
- Huang et al., *SnapMix: Semantically Proportional Mixing for Augmenting
  Fine-grained Data*, AAAI 2021, arXiv:2012.04846
- Michaeli et al., *Advancing Fine-Grained Classification by Structure and
  Subject Preserving Augmentation* (SaSPA), NeurIPS 2024, arXiv:2406.14551
- Wang et al., *Enhance Image Classification via Inter-class Image Mixup with
  Diffusion Model* (Diff-Mix), CVPR 2024
- Menon et al., *Long-tail learning via logit adjustment*, ICLR 2021
- OpenMixup, *Mixup Fine-grained and Scenic Classification Benchmarks*

> Ghi chú trung thực: hai file PDF (SnapMix bản AAAI, SaSPA bản NeurIPS) không
> trích xuất được bảng số chi tiết, nên **con số cụ thể của SnapMix trên NABirds
> chưa kiểm chứng được** — phải đọc lại từ bản gốc trước khi trích dẫn.
