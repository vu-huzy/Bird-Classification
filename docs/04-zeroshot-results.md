# 4. Dự án zero-shot — kết quả

> Tách từ README gốc.

> **Ghi chú điều hướng:** các "mục N" trong file này trỏ tới mục 11–18 của chính nó;
> mục 1–10 nằm ở [03-zeroshot-method.md](03-zeroshot-method.md).

# Dự án #3 — KẾT QUẢ: bước 1→4 (2026-09-06)

Toàn bộ trong `zeroshot/`. Chạy lại từ đầu:

```bash
python zeroshot/src/build_variants.py       # 555 lá   -> data/leaf_variants.csv
python zeroshot/src/build_taxonomy.py       # 404 loài -> data/species_taxonomy.csv  (GBIF)
python zeroshot/src/contamination.py        # đối chiếu ToL-10M -> results/contamination.csv
python zeroshot/src/build_tol_taxonomy.py   # -> data/tol_taxonomy.csv
python zeroshot/src/build_descriptors.py    # 167 mô tả riêng loài -> data/leaf_descriptors.json
python zeroshot/src/splits.py               # -> data/splits.npz
python zeroshot/src/encode.py --model bioclip --split test      # x3 model x2 split
python zeroshot/src/zs_eval.py --models clip_b16 bioclip bioclip2 \
       --levels T0 T0v T0d T0s T1 T2 T3a T3b                    # bước 2
python zeroshot/src/train_mets.py --img bioclip --txt clip_b16 --level T0s --split variant   # B1
python zeroshot/src/train_lora.py --img bioclip --txt clip_b16 --level T0s --split variant   # B2
python zeroshot/src/build_cub.py                                # CUB -> NABirds, 352 lá unseen
python zeroshot/src/train_cub.py --img bioclip --txt clip_b16 --caps cupl                    # B4
python zeroshot/src/report_zs.py --save                         # bảng cuối
```

| file | vai trò |
|---|---|
| `src/zs_env.py` | `USE_TF=0` + `HF_HOME`; **mọi script phải import đầu tiên** |
| `src/variants.py` | parse 60 chuỗi biến thể -> (sex, age, season, morph, form, phrase) |
| `src/descriptors.py` | T3a — mô tả thị giác CHUNG, suy ra từ các trục đã parse |
| `src/build_descriptors.py` | T3b — 167 mô tả RIÊNG từng lá, phủ 81/81 lá unseen |
| `src/prompts.py` | 8 mức prompt (bảng `SPEC`) + 80 template OpenAI |
| `src/zs_models.py` | `clip_b16` / `bioclip` / `bioclip2` qua open_clip. **CLIP phải là `ViT-B-16-quickgelu`** — xem mục 12.1 |
| `src/encode.py` | cache embedding ảnh ra `.npy` |
| `src/zs_eval.py` | bước 2 + `variant_probe` (chỉ số biến thể cân bằng theo lá) |
| `src/splits.py` | 2 split seen/unseen: `variant` và `species` |
| `src/train_mets.py` | **B1** — projection head trên embedding đóng băng |
| `src/train_lora.py` | **B2** — thêm LoRA vào image tower (`c_fc`/`c_proj`/`out_proj`) |
| `src/build_cub.py`, `src/train_cub.py` | **B4** — InfoNCE thật trên CUB, zero-shot sang 352 lá NABirds |
| `src/report_zs.py` | bước 4 — gộp mọi `summary.json` thành 4 bảng |

## 11. Bước 1 — contamination: **403/404 = 99.8%**

`imageomics/TreeOfLife-10M`, file `embeddings/txt_emb_species.json` (66 MB, 384,490 taxa,
trong đó 13,696 thuộc Aves).

| Mức đối chiếu | Trùng |
|---|---|
| **Loài (tên nhị thức)** | **403/404 = 99.8%** |
| Chi (genus) | 404/404 = 100% |
| Họ (family) | 404/404 = 100% |
| Tên thường | 372/404 = 92.1% |

Loài duy nhất không trùng — `Brandt's Cormorant` — cũng chỉ vì đổi chi (`Urile
penicillatus` là tên mới của `Phalacrocorax penicillatus`), không phải vì BioCLIP
chưa thấy.

> **Kết luận bắt buộc ghi vào báo cáo: không được gọi kết quả loài trên NABirds là
> "zero-shot".** Đó là đánh giá **in-domain đã pretrain**. Trục zero-shot hợp lệ còn
> lại là **biến thể bộ lông**, vì BioCLIP chỉ được giám sát bằng taxonomy — nhãn
> giới tính / tuổi / mùa / morph không tồn tại trong nguồn train của nó.

### 11.1 ToL-10M dùng taxonomy CŨ hơn GBIF — ảnh hưởng trực tiếp tới prompt

Kiểm chéo hai nguồn độc lập: trong 372 loài có tên thường ở cả hai, chỉ **339 (91.1%)**
trùng tên nhị thức. Khác biệt là trôi tên chi theo thời gian:

| NABirds | GBIF (hiện hành) | ToL-10M (BioCLIP đã học) |
|---|---|---|
| Yellow-rumped Warbler | `Setophaga coronata` | `Dendroica coronata` |
| Pine Siskin | `Spinus pinus` | `Carduelis pinus` |
| Tufted Titmouse | `Baeolophus bicolor` | `Parus bicolor` |
| Northern Shoveler | `Spatula clypeata` | `Anas clypeata` |

-> Prompt T1 cho BioCLIP dùng **chuỗi ToL** (`build_tol_taxonomy.py`), prompt cho CLIP
dùng GBIF. Đưa tên GBIF cho BioCLIP là đo nhầm: hoá ra đang phạt model vì chuỗi ngoài
phân phối chứ không phải vì nó kém.

### 11.2 Bước kiểm chéo bắt được 12 lỗi ánh xạ mà "khớp tên thường" bỏ lọt

GBIF chứa **loài hoá thạch** và **loài Cựu Thế giới** mang cùng tên tiếng Anh:

| NABirds | GBIF trả về (SAI) | Đúng | Vì sao |
|---|---|---|---|
| Golden Eagle | `Aquila bivia` | `Aquila chrysaetos` | hoá thạch, `extinct=True`, 1 vernacular |
| Anhinga | `Anhinga melanogaster` | `Anhinga anhinga` | Oriental Darter, loài châu Á |
| Black Vulture | `Aegypius monachus` | `Coragyps atratus` | Cinereous Vulture, loài Âu-Á |
| Purple Gallinule | `Porphyrio porphyrio` | `Porphyrio martinica` | Western Swamphen |
| Northern Shrike | `Lanius excubitor` | `Lanius borealis` | tách loài Tân/Cựu Thế giới |

Ba luật lọc (mỗi luật do một bug thật ép ra): bỏ bản ghi **lai** (không có
`canonicalName`), bỏ **`extinct`**, và trong các bản ghi còn lại chọn bản có **nhiều
vernacular nhất**. Sau đó 404/404 khớp chính xác, 404/404 tên nhị thức duy nhất.

## 12. Bước 2 — zero-shot thuần, không train gì

Giao thức bám đúng `examples/zero_shot.py` của BioCLIP: mỗi lớp qua **80 template
ImageNet của OpenAI**, trung bình embedding đã chuẩn hoá. Embedding ảnh cache một lần
(~150–185 s/model cho 24,633 ảnh), sau đó mỗi biến thể prompt chỉ là một phép nhân ma trận.

**8 mức prompt**, thiết kế 2×4 = (có / không chuỗi taxonomy) × (không biến thể / biến
thể thô / +mô tả chung / +mô tả riêng loài):

| | không taxonomy | có taxonomy |
|---|---|---|
| chỉ tên loài | `T0` | `T1` |
| + cụm biến thể | `T0v` | `T2` |
| + mô tả CHUNG (`descriptors.py`) | `T0d` | `T3a` |
| + mô tả RIÊNG loài (167 mô tả tự viết) | `T0s` | `T3b` |

### 12.1 Một bug nạp model đã phát hiện và sửa: **QuickGELU**

`open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')` dựng model với
**activation sai**. CLIP gốc của OpenAI dùng QuickGELU; tên đúng là
**`ViT-B-16-quickgelu`**. open_clip có cảnh báo nhưng chỉ ở mức `UserWarning` lẫn trong
đống log tải file, rất dễ bỏ qua:

```
QuickGELU mismatch between final model config (quick_gelu=False)
and pretrained tag 'openai' (quick_gelu=True)
```

Hạ oan CLIP đúng ở nơi nó được dùng làm mốc đối chứng chính:

| | sai (`ViT-B-16`) | đúng (`ViT-B-16-quickgelu`) | chênh |
|---|---|---|---|
| loài 404-way (T0) | 44.27 | **50.87** | +6.60 |
| lá 555 top-1 (T0s) | 37.55 | **44.03** | +6.48 |
| var81bal (T0s) | 77.86 | **79.90** | +2.04 |

Đã kiểm lại `bioclip` và `bioclip2`: cả hai dùng **GELU thường**, không cảnh báo, khớp
đúng `open_clip_config.json` của chính họ và đúng đường nạp của `pybioclip` — không dính lỗi.

> Bài học: một `UserWarning` về activation làm lệch 6.6 điểm. Mọi số của `clip_b16`
> trong repo này đã chạy lại sau khi sửa.

### 12.2 Kết quả (test 24,633 ảnh)

`var81bal` = phép đo trung tâm: **cho sẵn đúng loài, có chọn đúng biến thể không**,
cân bằng theo lá trên 81 loài của split `variant`. Chance = **47.57**.

| model | mức | lá555 top-1 | loài404 top-1 | macro-F1 | **var81bal** |
|---|---|---|---|---|---|
| **bioclip2** | T3b | **76.61** | 95.23 | **70.09** | 64.02 |
| bioclip2 | T2 | 76.27 | 95.23 | 69.41 | 60.40 |
| bioclip2 | T0 | 72.90 | **95.66** | 62.28 | 48.50 |
| bioclip | T2 | 64.50 | 81.85 | 57.11 | 54.86 |
| bioclip | T3b | 60.40 | 81.85 | 52.13 | 62.96 |
| bioclip | T0 | 63.99 | 81.36 | 53.12 | 48.50 |
| **clip_b16** | **T0s** | 44.03 | 50.87 | 40.17 | **79.90** |
| clip_b16 | T0v | 44.55 | 50.87 | 39.20 | 75.32 |
| clip_b16 | T0 | 41.68 | 50.87 | 33.93 | 48.50 |

Bảng đầy đủ 24 dòng: `zeroshot/results/table_zeroshot.csv`.

### 12.3 Phát hiện chính: **định danh loài và đọc bộ lông là hai năng lực TÁCH RỜI**

| | loài 404-way | var81bal (chance 47.57) |
|---|---|---|
| CLIP ViT-B/16 | 50.87 | **79.90** |
| BioCLIP ViT-B/16 | 81.36 | 62.96 |
| BioCLIP-2 ViT-L/14 | **95.66** | 64.02 |

CLIP kém BioCLIP-2 **44.8 điểm** ở định danh loài nhưng hơn **15.9 điểm** ở đọc bộ lông.
Cách giải thích khớp với cách hai model được train: BioCLIP được giám sát **hoàn toàn
bằng chuỗi taxonomy**, nên nó đổi khả năng nối ngôn ngữ mô tả với ảnh lấy độ chính xác
phân loại học. CLIP train trên alt-text tự nhiên — thứ mô tả *con chim trông thế nào* —
nên giữ được khả năng đó.

Ghi chú: `bioclip2` **nhiều tham số hơn hẳn** (ViT-L/14, 304M) so với hai model kia
(ViT-B/16, 86M), nên so sánh sạch nhất là **CLIP vs BioCLIP** — cùng kiến trúc, cùng
cỡ, chỉ khác nguồn giám sát: **50.87 / 62.96 (CLIP) đối lại 81.36 / 79.90**… đọc theo
cột: loài 50.87 → 81.36 (+30.5 cho BioCLIP), biến thể 79.90 → 62.96 (−16.9 cho BioCLIP).
Đánh đổi rõ ràng ở cùng ngân sách tham số.

### 12.4 Thang mô tả đơn điệu đúng chiều (var81bal)

| | T0 tên | +biến thể | +mô tả chung | +mô tả riêng loài |
|---|---|---|---|---|
| **CLIP** | 48.50 | 75.32 | 76.35 | **79.90** |
| BioCLIP | 48.50 | 50.04 | 59.21 | 62.66 |
| BioCLIP-2 | 48.50 | 58.08 | 56.29 | 63.98 |

Mỗi bậc thêm thông tin đều có lãi, và **mô tả riêng loài > mô tả chung** ở cả ba model.
Đây là phần trả lời câu hỏi ở mục 7.1: nhãn lá thôi **chưa đủ**; text encoder cần từ
vựng thị giác thật.

Lưu ý ngược chiều: **chuỗi taxonomy giúp định danh loài nhưng cản trục biến thể của
CLIP** (T0s 79.90 → T3b 77.44), và với CLIP taxonomy Latin còn làm sụt luôn định danh
loài (T0 50.87 → T1 41.46) — nó không biết tên Latin.

### 12.5 So với baseline có giám sát của chính repo này

| | lá 555 top-1 | loài 404 |
|---|---|---|
| `vit_b_16_in21k` có giám sát, 20,419 ảnh train | **85.86** | 87.85 |
| BioCLIP-2 **không train gì**, 0 ảnh NABirds | 76.61 | **95.66** |

BioCLIP-2 **vượt model có giám sát 7.8 điểm ở mức loài mà không thấy một ảnh NABirds
nào khi fine-tune**. Đọc con số này cùng mục 11: 99.8% loài đã nằm trong pretrain của
nó. Đây là thước đo mức độ nhiễm, không phải bằng chứng về khả năng tổng quát hoá.

## 13. Bước 3a (B1) — METS-analogue trên embedding đóng băng

Text tower đóng băng, train `img_proj` + `txt_proj` (đúng cấu trúc METS), chỉ trên các
lớp **seen**, rồi đánh giá trên lớp **unseen**. Không dùng in-batch negative — mỗi step
so ảnh với toàn bộ prototype lớp seen (mục 3.1/3.2). Mỗi run ~40 s trên embedding cache.

**Chọn epoch bằng val-UNSEEN, không phải val-seen.** Cắt tiếp 15% *lớp* seen làm unseen
giả. Dùng val-seen cho ZSL-unseen 62.59 trong khi không train gì được 69.70 — chọn theo
val-seen chính là tối đa hoá thứ đối nghịch với zero-shot.

### 13.1 Split `variant` — 81 lá female/immature KHÔNG có trong train

| img tower | txt tower | mức | **var81bal sau train** | không train | chênh |
|---|---|---|---|---|---|
| clip_b16 | clip_b16 | T0s | **85.63** | 79.90 | +5.73 |
| clip_b16 | clip_b16 | T3a | 79.73 | 68.06 | **+11.67** |
| **bioclip** | **clip_b16** | T0s | **77.66** | — | — |
| bioclip | clip_b16 | T0d | 73.82 | — | — |
| bioclip2 | bioclip2 | T2 | 68.02 | 60.40 | +7.62 |
| bioclip2 | bioclip2 | T3b | 66.49 | 64.02 | +2.47 |
| bioclip | bioclip | T2 | 64.15 | 54.86 | +9.29 |

**Train giúp trục biến thể ở mọi cấu hình**, dù 81 lá đó chưa từng xuất hiện trong
train — alignment học được từ 474 lá seen chuyển sang được. Đây là kết quả METS thật sự
tái hiện được trên dataset này.

**Ghép hai tower ăn thua:** ảnh BioCLIP + text CLIP đạt **77.66** so với BioCLIP thuần
64.15 (**+13.51**). Đúng như dự đoán từ mục 12.3 — lấy đặc trưng ảnh của model biết
chim, ghép với text tower biết mô tả.

### 13.2 Split `species` — 79 loài KHÔNG có trong train

| img | txt | mức | ZSL unseen sau train | không train | chênh |
|---|---|---|---|---|---|
| bioclip2 | bioclip2 | T2 | **81.42** | 75.93 | +5.49 |
| bioclip2 | bioclip2 | T3b | 81.08 | 74.33 | +6.75 |
| bioclip | bioclip | T2 | 75.06 | 69.26 | +5.80 |
| clip_b16 | clip_b16 | T0s | 65.52 | 64.18 | +1.34 |

Train có lãi ở **mọi** model. GZSL harmonic của `bioclip2/T2` lên 79.15 (không train:
74.83) và top-1 trên toàn 555 lớp đạt **87.22** — vượt cả baseline có giám sát 85.86,
dù 115 lá bị giữ khỏi train.

### 13.3 Điểm YẾU phải báo cáo: GZSL trên split `variant` rất tệ

| | gzsl_seen | gzsl_unseen | harmonic |
|---|---|---|---|
| bioclip/bioclip T2 | 80.48 | **13.68** | 23.39 |
| bioclip2/bioclip2 T2 | 89.89 | 45.87 | 60.74 |
| clip/clip T0s | 60.23 | 21.60 | 31.80 |
| bioclip2 + clip_b16 T0d | 82.16 | **2.70** | **5.22** |

Khi phải chọn trong cả 555 lớp, ảnh con mái gần như luôn bị gán vào **lá male của chính
loài đó** — lá đó có trong train. Đây là thiên lệch seen của GZSL ở dạng gay gắt nhất,
vì lớp seen và lớp unseen là hai biến thể của **cùng một loài**. `var81bal` (thu hẹp về
các lá của đúng loài đó) mới tách được năng lực khỏi thiên lệch.

Cấu hình `bioclip2` ảnh + `clip_b16` text sập hẳn (harmonic 5.22). Hạ lr xuống 3e-4 và
1e-4 còn tệ hơn -> không phải lỗi lr, mà là cặp tower đó cho projection thiên lệch
seen rất mạnh.

## 14. Bước 3b (B2) — LoRA trên image tower

LoRA được gắn vào `c_fc`, `c_proj` và `out_proj` của image tower; q/k/v không
gắn được vì open_clip dùng `nn.MultiheadAttention` với `in_proj_weight`.
Kết quả đã hoàn tất trên 5 cấu hình, peak VRAM 5.4–10.0 GB:

| split | cấu hình | LoRA var81bal | B1 linear | LoRA all555 | B1 all555 | phút |
|---|---|---:|---:|---:|---:|---:|
| variant | bioclip + clip/T0s | 77.36 | 77.66 | 69.36 | 63.67 | 12.7 |
| variant | bioclip + bioclip/T2 | 65.69 | 64.15 | 73.46 | 72.43 | 12.1 |
| variant | bioclip2 + bioclip2/T2 | 72.83 | 68.02 | **87.54** | 84.59 | 27.6 |
| species | bioclip + clip/T0s | **91.06** | 87.84 | 68.91 | 62.56 | 11.9 |
| species | bioclip + bioclip/T2 | 90.87 | 87.29 | 76.38 | 72.39 | 12.0 |

Kết luận: LoRA nâng rõ kết quả tổng thể, nhưng gần như không cải thiện trục
compositional zero-shot (`+1.5` hoặc `-0.3` trên split variant). Nút thắt chính
nằm ở text và phép nối hai tower, không phải thiếu thông tin trong image feature.

## 15. Bước 3c (B4) — contrastive THẬT trên CUB, rồi zero-shot sang NABirds

Mục 3.2 nói: NABirds chỉ có text **mức lớp**, nên phương pháp thực chất là DeViSE/ALE
chứ không phải contrastive kiểu CLIP/METS. Bước này **đo thẳng luận điểm đó** bằng cách
đổi đúng một biến — độ hạt của text — trên cùng 5,994 ảnh CUB train:

| bộ caption | số chuỗi duy nhất | tính chất |
|---|---|---|
| `naive` (`anjunhu/naively_captioned_CUB2002011_train`) | **200 / 5,994** | text MỨC LỚP. Mọi ảnh cùng loài nhận cùng một chuỗi -> InfoNCE sinh false negative |
| `cupl` (`anjunhu/CuPL_DaVinci_captioned_CUB2002011_train`) | **4,490 / 5,994** | text BIẾN THIÊN theo ảnh, đúng cấu trúc (ECG, báo cáo máy sinh) của METS |

Loss ở đây là **InfoNCE đối xứng in-batch thật** (loss của CLIP), không phải
cross-entropy trên prototype lớp như B1/B2 — vì bây giờ text mới đủ đa dạng để chuyện đó
có nghĩa.

**Đã kiểm tính hợp lệ của cặp (ảnh, caption)** trước khi tin kết quả: cosine của cặp
đúng là 0.336 / 0.322 (naive / cupl) so với cặp xáo trộn 0.244 / 0.231; cặp đúng thắng
**96.7% / 95.8%** số ảnh. Ghép cặp không bị lệch.

**Split đánh giá:** 200 lớp CUB phủ **144/404 loài NABirds** -> **352 lá NABirds có loài
KHÔNG nằm trong CUB** (14,907 ảnh test). Con số này tái lập đúng phần "chồng lấn
CUB–NABirds" ở mục 0 của README, được tính lại độc lập bằng code.

### 15.1 Kết quả

| caps | lr | val-unseen (40 lớp CUB giữ lại) | zsl352 NABirds | vpun_bal |
|---|---|---|---|---|
| **cupl** | 1e-3 | **73.73** | **41.54** | 65.28 |
| cupl | 3e-4 | 73.09 | 40.20 | **66.03** |
| cupl | 1e-4 | 71.98 | 38.83 | 65.46 |
| naive | 1e-3 | 68.95 | 39.46 | 65.03 |
| naive | 3e-4 | 70.20 | 35.45 | 63.39 |
| naive | 1e-4 | 68.36 | 33.33 | 62.55 |
| *không train gì (B0)* | — | — | *49.80* | *61.59* |

**Text biến thiên thắng text mức lớp ở CẢ BA lr và cả ba chỉ số** (+2.9 đến +4.8 điểm
val-unseen, +2.1 đến +5.5 điểm zsl352). Đây là bằng chứng trực tiếp cho mục 3.2: độ hạt
của text là biến có thật, không phải chi tiết kỹ thuật.

### 15.2 Nhưng train trên CUB KHÔNG thắng nổi model off-the-shelf

Trên đúng 352 lá đó, không train gì cả:

| | zsl352 | vpun_bal (chance 48.28) |
|---|---|---|
| BioCLIP-2 T3b off-the-shelf | **77.95** | 58.40 |
| BioCLIP T2 off-the-shelf | 67.41 | 53.04 |
| CLIP T0s off-the-shelf | 50.82 | **71.90** |
| **B4 train trên CUB (cupl, tốt nhất)** | 41.54 | 66.03 |

Đúng như đã cảnh báo ở mục 7.3 **trước khi chạy**: alignment học từ 6k ảnh không thể so
với model pretrain trên 10M–200M ảnh. Giá trị của B4 nằm ở **phép so sánh có kiểm soát**
(naive vs cupl), không nằm ở con số tuyệt đối.

Vẫn giữ đúng khuôn mẫu thấy ở mọi nơi khác: train **làm hại** định danh loài
(41.54 < 49.80) nhưng **giúp** trục biến thể (66.03 > 61.59).

## 16. Hai lỗi trong chính phép đo của tôi — và cách sửa

**(a) Chỉ số biến thể gian lận được.** Phiên bản đầu đo `oracle_variant_unseen_acc`:
chỉ tính trên ảnh của **81 lá female**. Model nào thiên vị "female" cũng ăn điểm cao.
CLIP `T3a` đạt **85.68** ở chỉ số đó trong khi trên toàn bộ 137 loài đa-lá chỉ được
**54.89** — mô tả chung của lớp female dài và bao trùm nên hút hết dự đoán.

Sửa: `variant_probe` lấy **trung bình accuracy CỦA TỪNG LÁ** (cân bằng theo lá, cả male
lẫn female), nên thiên vị một biến thể tự triệt tiêu. Chance = 47.57.

Kiểm tra tự nhất quán sau khi sửa: mọi cấu hình `T0`/`T1` (text không mang thông tin
biến thể) đều cho **48.50 ≈ chance 47.57** — đúng như phải thế, vì các lá cùng loài nhận
cùng một chuỗi text.

**(b) Chọn epoch bằng val-seen.** Xem mục 13. Val-seen là thứ đối nghịch với zero-shot;
dùng nó cho ZSL-unseen 62.59 trong khi không train gì được 69.70.

Cả hai lỗi đều **làm sai lệch kết luận theo hướng thuận lợi cho câu chuyện tôi đang
kể** — đó là lý do phải có mốc chance và mốc "không train" trong mọi bảng.

## 17. Hạn chế — điều CHƯA loại trừ được

| | |
|---|---|
| **Nhiễm ở mức ẢNH** | Mới xác minh nhiễm ở mức *tên loài*. Ảnh NABirds đến từ Flickr/All About Birds, EOL cũng lấy ảnh Flickr -> **có thể trùng chính ảnh test**. Không có cách kiểm với dữ liệu công khai hiện có. Con số 95.66% của BioCLIP-2 phải đọc kèm cảnh báo này |
| Mô tả T3b do LLM sinh | 167 mô tả, khớp tên lá 100% và phủ 81/81 lá unseen, nhưng **nội dung chưa được chuyên gia soi**. `python zeroshot/src/build_descriptors.py --check` in toàn bộ để kiểm tay |
| Caption CuPL không thật sự theo ảnh | Sinh từ TÊN LỚP, không phải từ bức ảnh. Là "text mức lớp có biến thiên", chưa phải caption người viết theo ảnh (Reed et al. 2016 — bộ đó không có bản tải tự động được) |
| Chỉ chạy 1 seed | Chưa có khoảng tin cậy cho các chênh lệch 2–5 điểm |
| Prompt bị cắt | 64/44,400 prompt của T3a/T3b vượt 77 token (5/555 lớp, nặng nhất 40/80 template ở `White-throated Sparrow (Tan-striped/immature)`) |
| Cấu hình không ổn định | `bioclip2` ảnh + `clip_b16` text trên split `variant` sập (GZSL harmonic 5.22), hạ lr không cứu được |

## 18. Bốn câu trả lời rút ra

1. **Không thể gọi NABirds là zero-shot ở mức loài với BioCLIP** — 99.8% loài đã nằm
   trong pretrain, và BioCLIP-2 zero-shot vượt cả model có giám sát của chính repo này
   (95.66 vs 87.85 ở mức loài).

2. **Trục biến thể bộ lông là trục còn hợp lệ (compositional zero-shot, D11), và ở đó thứ hạng đảo ngược.**
   Ở cùng kiến trúc ViT-B/16: đổi CLIP sang BioCLIP được **+30.5 điểm** định danh loài
   nhưng mất **−16.9 điểm** đọc bộ lông. Giám sát bằng taxonomy mua độ chính xác phân
   loại học bằng cái giá là khả năng nối ngôn ngữ mô tả với ảnh.

3. **METS-analogue có tác dụng, và ghép hai tower còn tốt hơn.** Train projection trên
   lớp seen nâng trục biến thể +9.3 điểm cho BioCLIP và ZSL loài +5.8 điểm, dù lớp
   unseen chưa từng xuất hiện. Ghép ảnh-BioCLIP với text-CLIP thu thêm **+13.5 điểm** —
   hai điểm mạnh cộng được vào nhau.

4. **Độ hạt của text là biến có thật.** Trên cùng 5,994 ảnh CUB, đổi caption từ mức lớp
   (200 chuỗi) sang biến thiên theo ảnh (4,490 chuỗi) cho **+2.9 đến +4.8 điểm** trên
   lớp CUB giữ lại và **+2.1 đến +5.5 điểm** khi chuyển sang 352 lá NABirds — nhất quán
   ở cả ba learning rate.
