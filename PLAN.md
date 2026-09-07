# PLAN — kế hoạch & theo dõi tiến độ

Cập nhật: 2026-09-07. File này trả lời ba câu: **đang ở đâu**, **đã quyết gì và vì sao**,
**làm gì tiếp**. Kết quả chi tiết ở [README.md](README.md) và [`docs/`](docs/).

---

## 1. Trạng thái tổng thể

| Dự án | Trạng thái | Ghi chú |
|---|---|---|
| 1 — Binary classification | ⛔ **bỏ** | Quyết định **D13** — không cho kết luận nào mà bài 555 lớp chưa có |
| 2 — Multi-class 555 lớp | ✅ **xong** | **44 run qua 8 pass**; kỷ lục **90.03%** |
| 3 — Zero-shot CLIP/METS | ✅ **xong** | mọi việc kiểm chứng đã hoàn tất |

### Dự án 2 — chi tiết

| # | Việc | Trạng thái |
|---|---|---|
| 2.1 | Pre-resize 48k ảnh → `images_r448` | ✅ 9.4 GB → 3.3 GB, epoch nhanh 8x |
| 2.2 | CNN tự xây (baseline không pretrain) | ✅ 60.72 @224 / 63.74 @448 |
| 2.3 | ResNet-50/101, Inception-v3, ViT-B/16 (IN1k + IN21k) | ✅ tốt nhất 85.86 |
| 2.4 | Đối chứng độ phân giải 224 vs 448 | ✅ resnet50 **+4.57**, ViT **+4.17** |
| 2.5 | Phân tích lỗi (per-class / per-species / per-order / confusions) | ✅ |
| 2.6 | **LoRA thay cho full fine-tune** | ✅ 85.06 với **3.2% tham số** nhưng **chậm hơn 38%** |
| 2.7 | Module FGVC chuyên dụng (PMG / WS-DAN / TransFG) | ⬜ chưa làm — nhưng xem 2.16 |
| 2.8 | NABirds → fine-tune CUB | ✅ xem 2.17 |
| 2.9 | **CNN hiện đại** (ConvNeXt-T, EfficientNetV2-S) + cầu nối resnet50-AdamW | ✅ ConvNeXt-T IN22k **85.55** với 1/3 tham số của ViT |
| 2.10 | **Trục pretrain điểm thứ 4**: BioCLIP (ToL-10M) | ✅ **87.72** — IN1k→sinh học +8.53, nhưng +6.67 đã lấy được bằng IN21k |
| 2.11 | **Trục kiến trúc/quy mô**: VGG16, DenseNet, MobileNetV3, ResNeXt, Swin-T, ViT-S | ✅ ConvNeXt-T > ViT-S > Swin-T ở cùng ~25M/IN21k |
| 2.12 | **C4 — ViT-B/16 IN21k @448** | ✅ **90.03 — kỷ lục repo**, cách TransFG 0.77 điểm |
| 2.13 | **Ensemble + TTA** (không train gì) | ✅ TTA +0.2…+1.71; ensemble **âm** (val quá nhỏ để xếp hạng) |
| 2.14 | **Trục thời gian 2012→2022**: AlexNet, GoogLeNet, EfficientNet-B0 | ✅ +43.2 điểm tổng, nhưng 2014→2022 chỉ +6.4 |
| 2.15 | **Logit adjustment hậu kiểm** (không train gì) | ✅ **+5.78 F1 lớp khó**, miễn phí |
| 2.16 | **Part-query decoder** (encoder-decoder + cross-attention) | ✅ **encoder đóng băng: +1.12 top-1, +4.23 recall lớp khó** — xem mục E.5 |
| 2.17 | **C3 — NABirds → CUB** | ✅ **3/3 model đều dương**: bioclip +0.28, ConvNeXt +0.31, ViT +0.17 (TB +0.25) — nhất quán nhưng nhỏ |
| 2.18 | **Augmentation** (6a: 6 run ablation, 6b: 12 run CMO) | ✅ xem mục E |

### Dự án 3 — chi tiết

| # | Bước | Trạng thái | Kết quả |
|---|---|---|---|
| 3.0 | Hạ tầng text: parser biến thể + taxonomy Latin | ✅ | 555/555 lá parse được; 404/404 loài khớp GBIF |
| 3.1 | Contamination audit vs TreeOfLife-10M | ✅ | **403/404 = 99.8%** → đổi trục |
| 3.2 | Zero-shot thuần, 3 model × 10 mức prompt | ✅ | 24 run; hai năng lực tách rời |
| 3.3 | B1 — projection head trên embedding đóng băng | ✅ | 22 run |
| 3.4 | B2 — LoRA trên image tower | ✅ | 5 run; VRAM 5.4–10.0 GB |
| 3.5 | B4 — InfoNCE thật trên CUB (naive vs CuPL) | ✅ | 6 run; text biến thiên thắng ở cả 3 lr |
| 3.6 | Trần tham chiếu từ model có giám sát | ✅ | **vp81_bal = 95.61** |
| 3.7 | Đối chứng hoán vị mô tả | ✅ | tụt **9.4–13.2 điểm** → nội dung mang thông tin thật |
| 3.8 | 3 seed cho các so sánh chính | ✅ | sd 0.29–1.64; `bioclip2`/species dao động **±6.73** |
| 3.9 | B3 — full fine-tune image tower | ⬜ bỏ | B2 cho thấy nút thắt không ở phía ảnh |

---

## 2. Thang đo của dự án 3 — đọc mọi con số biến thể theo thang này

Trục biến thể (`var81bal`, 81 loài, cân bằng theo lá). **Thuật ngữ đúng:
compositional zero-shot** (D11) — model đã thấy ẢNH chim mái, chỉ chưa thấy NHÃN.

```
chance                                    47.57
BioCLIP off-the-shelf, prompt tốt nhất    62.96   ── phục hồi 32% khoảng cách
CLIP off-the-shelf, prompt tốt nhất       79.90   ── phục hồi 67%
METS-analogue (clip+clip, T0s)            85.63   ── phục hồi 79%
model CÓ GIÁM SÁT (trần)                  95.61   ── 100%
```

---

## 3. Quyết định đã chốt (và vì sao)

| # | Quyết định | Lý do | Ngày |
|---|---|---|---|
| D1 | Dùng nguyên `train_test_split.txt`, **không** dùng bbox lúc test | giao thức của TransFG / MetaFormer / MPSA | 09-06 |
| D2 | Chỉ gộp nhãn ở 2 mức: species (404) và order (22) | cây không đồng đều độ sâu | 09-06 |
| D3 | `RandomResizedCrop(scale=(0.3,1.0))`, hue jitter ≈ 0 | bbox chim median 28.3% diện tích; **màu bộ lông CHÍNH LÀ nhãn** | 09-06 |
| D4 | Không dùng weighted sampler | imbalance nhẹ (mean 43.1 ảnh/lớp) | 09-06 |
| D5 | bf16 chứ không fp16 | RTX 5070 là Blackwell sm_120 | 09-06 |
| D6 | **Đổi trục zero-shot từ loài sang biến thể bộ lông** | 99.8% loài đã nằm trong pretrain của BioCLIP | 09-06 |
| D7 | Prompt cho BioCLIP dùng taxonomy **của ToL-10M**, không phải GBIF | 33/404 loài khác tên chi | 09-06 |
| D8 | Bỏ in-batch negative ở B1/B2 | text mức lớp → InfoNCE in-batch sinh false negative | 09-06 |
| D9 | Chọn epoch bằng **val-UNSEEN** | val-seen là thứ đối nghịch với zero-shot | 09-06 |
| D10 | Chỉ số biến thể phải **cân bằng theo lá** | bản tính theo ảnh gian lận được | 09-06 |
| D11 | Gọi đúng tên: **compositional zero-shot** | model đã thấy ẢNH chim mái, chỉ chưa thấy NHÃN | 09-06 |
| D12 | CLIP phải nạp bằng `ViT-B-16-quickgelu` | nạp sai activation hạ oan CLIP 6.6 điểm | 09-06 |
| D13 | **Bỏ hẳn dự án 1 (phân loại nhị phân)** | không cho kết luận nào mà bài 555 lớp chưa có | 09-07 |
| D14 | ConvNeXt / EfficientNetV2 / Swin dùng **AdamW**; ViT và CNN BatchNorm dùng **SGD** | probe 4 epoch: ConvNeXt + lr SGD của ViT tụt **49 điểm** | 09-07 |
| D15 | `--img-size` phải truyền vào **lúc dựng model**, không chỉ vào transform | ViT gắn pos-embed với số patch | 09-07 |
| D16 | **Không làm augmentation sinh ảnh** (SaSPA / Diff-Mix / DiffuseMix) | SaSPA đo CUB **+0.7** — thấp nhất trong 5 dataset — vì *class-fidelity thấp*; cần 4× RTX 3090; rủi ro sinh ảnh SAI loài | 09-07 |
| D17 | **Mọi chỉ số augmentation phải tách theo nhóm đuôi / nhóm khó** | 113 lớp đuôi chỉ chiếm ~10% ảnh test nên top-1 tổng che mất hiệu ứng thật | 09-07 |
| D18 | **Tập "lớp khó" định nghĩa bằng F1 trung bình trên TOÀN BỘ run** | lấy từ một model rồi so chính model đó là selection bias — phóng đại ~18 lần | 09-07 |
| D19 | **Một thay đổi augmentation phải chạy trên ≥3 model trước khi kết luận** | E1 đọc ra ba kết luận khác nhau ở 1, 2 và 4 model. Chỉ số nhóm nhỏ (113/20 lớp) nhiễu hơn top-1 nhiều | 09-07 |

---

## 4. Việc còn lại, xếp theo giá trị

### Đã hoàn thành trong hàng đợi (09-07)

| việc | kết quả |
|---|---|
| ~~C4 — ViT-B/16 IN21k @448~~ | ✅ **90.03**, +4.17 do độ phân giải, cách TransFG **0.77** |
| ~~BioCLIP cho bài có giám sát~~ | ✅ **87.72**, nhưng lợi ích dồn hết vào lớp đầu |
| ~~Ensemble + TTA~~ | ✅ TTA +0.2…+1.71; ensemble **âm** |
| ~~Trục thời gian kiến trúc~~ | ✅ 2012→2022 = +43.2, nhưng 2014→2022 chỉ +6.4 |
| ~~A3 — thuật ngữ compositional ZS~~ | ✅ hộp thuật ngữ ở README + docs/03 |
| ~~C5 — lớp đuôi~~ | ✅ logit adjustment hậu kiểm: **+5.78 F1 lớp khó, miễn phí** |
| ~~C3 — NABirds → CUB~~ | ✅ **3/3 model dương** (TB +0.25). Nhỏ đúng như dự đoán: 142/200 loài CUB đã có trong NABirds nên không phải miền mới |
| ~~Part-query decoder~~ | ✅ **+1.12 top-1, +4.23 recall lớp khó** với encoder ĐÓNG BĂNG |
| ~~E — augmentation~~ | ✅ 18 run; xem mục E |

### Còn lại

| # | việc | vì sao đáng | chi phí |
|---|---|---|---|
| **P1** | **Sửa nhánh fine-tune của part-query** | `partq_ft` chỉ đạt 78.18 so với 86.98 của bản đóng băng — AdamW lr 1e-4 phá đặc trưng ViT vốn train bằng SGD. Đúng bẫy #12. Sửa lr rồi chạy lại, kỳ vọng vượt 86.98 | ~1h |
| **P2** | **Part-query trên encoder @448** | encoder 448 cho **784 token** thay vì 196 ⇒ bản đồ bộ phận mịn hơn hẳn. Encoder tốt nhất (90.03) + decoder | ~2h |
| **P3** | **Đối chiếu attention với 11 keypoint NABirds** | phép đo mà tài liệu hầu như chỉ làm trên CUB. Dữ liệu đã có sẵn (`nabirds/parts`) | ~2h |
| C2 | Module FGVC (bilinear → WS-DAN → PMG) | part-query đã cho +1.12; WS-DAN là hướng gần nhất còn lại | ~3h |
| C6 | Đa nhiệm species × biến thể | `variants.py` đã parse sẵn; nhắm `err_same_species` 17.4% | ~2h |
| A2 | Kiểm nhiễm ở **mức ảnh** | rủi ro lớn nhất chưa loại trừ — áp cho cả 87.72 của BioCLIP | cao |
| B2 | Chuyên gia soi 167 mô tả T3b | cần người có chuyên môn | — |

### Đã bỏ, có lý do

| việc | vì sao bỏ |
|---|---|
| C1 — dự án 1 (nhị phân) | **D13** |
| B1 — calibrated stacking GZSL | bài toán bị đặt sai (seen/unseen là hai biến thể cùng loài), không phải thiếu calibration |
| E5 — augmentation sinh ảnh | **D16** |
| B3 — full fine-tune image tower (dự án 3) | B2 cho thấy nút thắt không ở phía ảnh |

---

## E. Augmentation — kết quả đầy đủ

Chi tiết + nguồn: [`docs/02` § Pass 6](docs/02-supervised.md).

### E.1 Ablation trên `resnet50` @224 (pass 6a)

| cấu hình | top-1 | F1@113 lớp đuôi | recall@20 lớp khó | kết luận |
|---|---|---|---|---|
| baseline (30 ep) | 78.61 | 62.04 | 22.70 | — |
| `--erasing-p 0` | 78.69 | 63.18 | 24.41 | ✅ nhỏ nhưng thật (4/4 model) |
| `--rotate 15` | 78.50 | **63.29** | **25.43** | ✅ giữ |
| `--hue 0.10` | **77.92** | **61.91** | 23.40 | ❌ bỏ — xác nhận D3 |
| 100 epoch | **80.25** | **64.30** | 22.37 | ✅ đòn bẩy lớn nhất |
| 100 epoch + CutMix | **80.48** | 63.87 | **24.11** | ✅ dương nhưng nhỏ |

**Ngân sách epoch (+1.64) lớn hơn cả grouped conv (+1.08) và nhân đôi độ sâu
(+0.95)** ⇒ mọi run CNN 30-epoch của repo đang **thiếu train ~1.6 điểm**.

**D3 xác nhận bằng số**: hue 0.10 làm hại **gấp 2.4 lần** ở 77 lớp màu-là-nhãn
(Warbler/Tanager/Oriole/Finch/Bunting: −1.22 F1) so với 478 lớp còn lại (−0.50).

**E1 — phải chạy 4 model mới nói đúng được:**

| model | Δ top-1 | Δ F1@đuôi | Δ recall@khó |
|---|---|---|---|
| `resnet50` | +0.08 | +1.14 | +1.71 |
| `convnext_tiny_in22k` | +0.13 | −0.49 | −0.45 |
| `vit_b_16_in21k` | +0.33 | +0.61 | −2.35 |
| `bioclip` | +0.16 | +1.51 | +2.00 |

**Top-1 dương 4/4** (trung bình +0.175) — bốn lần cùng dấu thì xác suất ngẫu
nhiên là 1/16 ⇒ hiệu ứng **thật nhưng nhỏ hơn nhiễu của từng run**. Nhóm
đuôi/khó **không** nhất quán (3/4 và 2/4).

Bài học: sau 1 model tôi tưởng đã xác nhận; sau 2 model tôi kết luận "không nhất
quán"; phải đủ **4** mới thấy đúng — nhất quán ở top-1, nhiễu ở chỉ số nhóm nhỏ.
`bioclip_224_noerase` **87.88** là model đơn tốt thứ hai của repo.

### E.2 CMO — lợi ích tỉ lệ NGHỊCH với độ mạnh của model

CMO (Park et al., CVPR 2022) dán chim của 113 lớp hiếm (pool 2,377 ảnh) lên nền
của lớp nhiều ảnh. Bản của repo dùng **bbox thật** thay vì ô ngẫu nhiên.

**Trên `resnet50` (model YẾU, 78.61) — có tác dụng, và có điểm tối ưu:**

| `cmo-p` | top-1 | F1@đuôi | recall@khó |
|---|---|---|---|
| 0 | 78.69 | 63.18 | 24.41 |
| 0.25 | **78.73** | 64.50 | 25.80 |
| 0.50 | 78.25 | **64.71** | **28.28** |
| 0.75 | 77.36 | 63.31 | 25.87 |

Đơn điệu tăng tới 0.50 rồi quay đầu ⇒ **có quan hệ liều–đáp ứng thật**, không
phải nhiễu. Ở 0.50: **+5.58 recall lớp khó** so với baseline gốc.

**Trên model MẠNH (cmo-p 0.5, so với chính bản `_noerase` của nó) — HẠI:**

| model | Δ top-1 | Δ F1@đuôi | Δ recall@khó |
|---|---|---|---|
| `convnext_tiny_in22k` | −0.44 | +0.61 | −0.16 |
| `vit_b_16_in21k` | −1.00 | −1.48 | −1.72 |
| `bioclip` | **−1.17** | **−5.51** | **−10.98** |

> **Kết luận quan trọng nhất của mục E.** Giá trị của CMO **tỉ lệ nghịch với độ
> mạnh của model** — đúng dạng đã thấy ở TTA. Model yếu được lợi từ bối cảnh đa
> dạng thêm; model mạnh vốn đã xử lý được lớp đuôi, nên ảnh ghép chỉ còn là
> **nhiễu nhãn** mà nó đủ sức khớp vào và hỏng theo. BioCLIP hỏng nặng nhất
> (−10.98 recall lớp khó), hợp lý vì nó là model có đặc trưng chuyên biệt nhất.
>
> ⇒ **Không áp CMO cho model mạnh.** Đây là kết quả âm và phải ghi lại như vậy.

### E.3 Cộng dồn: không

`resnet50` + `--rotate 15`: một mình cho +1.25 F1 đuôi. Kết hợp với `_noerase`
cho 63.45 (thấp hơn `_noerase` + CMO 0.25 = 64.50), và kết hợp cả ba
(`_noerase + rot15 + cmo50`) cho 63.33 — **thấp hơn CMO 0.50 một mình (64.71)**.
Các phép augmentation này **đạp nhau chứ không cộng dồn**.

### E.4 Bảng tổng kết augmentation

| thay đổi | bằng chứng | kết luận |
|---|---|---|
| Ngân sách epoch 30→100 | +1.64 top-1, +2.26 F1 đuôi | ✅ đòn bẩy lớn nhất |
| CMO trên model YẾU | liều–đáp ứng đơn điệu, +5.58 recall khó | ✅ dùng cho model yếu |
| Xoay 15° | +1.25 đuôi, +2.73 khó (top-1 −0.11) | ✅ 1 model, cần lặp lại |
| CutMix @100ep | +0.23 top-1, +1.74 khó | ✅ dương nhưng nhỏ |
| Tắt RandomErasing | top-1 +0.175 trung bình, **dương 4/4 model** | ✅ thật nhưng rất nhỏ |
| CMO trên model MẠNH | bioclip −5.51 đuôi, −10.98 khó | ❌ **có hại** |
| Hue 0.10 | hại gấp 2.4× ở lớp màu-là-nhãn | ❌ bỏ — xác nhận D3 |
| Diffusion sinh ảnh | SaSPA +0.7 trên CUB, thấp nhất/5 dataset | ❌ bỏ không chạy (D16) |

### E.5 Part-query decoder — kết quả dương mạnh nhất còn lại

Encoder-decoder + cross-attention, K = 12 part query, decoder 15.14M tham số.

| run | encoder | top-1 | F1@đuôi | recall@khó |
|---|---|---|---|---|
| **`partq_frozen`** | ViT-B/16 IN21k đã fine-tune NABirds, **ĐÓNG BĂNG** | **86.98** | **74.78** | **37.51** |
| *(encoder của chính nó)* | `vit_b_16_in21k_224` | 85.86 | 72.48 | 33.28 |
| **chênh** | | **+1.12** | **+2.30** | **+4.23** |
| `partq_ft_k24` | như trên, mở khoá, K=24 | 81.68 | 66.74 | 26.80 |
| `partq_ft` | như trên, mở khoá, K=12 | 78.18 | 62.73 | 22.12 |
| `partq_imagenet` | ImageNet (đối chứng kiểu INTR) | 75.05 | — | — |

Ba điều đọc được:

1. **Chỉ train 15.14M tham số decoder trên encoder ĐÓNG BĂNG cho +1.12 top-1 và
   +4.23 recall lớp khó.** Đây là mức tăng lớn nhất mà một module thêm vào đạt
   được trong repo.
2. **Nhánh fine-tune HỎNG** (78.18 so với 86.98). Nguyên nhân gần như chắc chắn
   là AdamW lr 1e-4 phá đặc trưng của ViT vốn được train bằng SGD — đúng **bẫy
   #12**. Đây là lỗi kỹ thuật sửa được, không phải thất bại của ý tưởng → **P1**.
3. **Đối chứng kiểu INTR đúng như dự đoán**: khởi tạo từ ImageNet cho 75.05,
   kém bản đóng băng **11.93 điểm**. Khớp với việc INTR chỉ đạt CUB 71.8% so với
   83.8% của ResNet-50 thường — họ dùng backbone yếu.

---

## 5. Rủi ro đã biết (chưa loại trừ được)

| Rủi ro | Ảnh hưởng | Trạng thái |
|---|---|---|
| Nhiễm ở mức **ảnh** | mọi con số BioCLIP/BioCLIP-2 có thể bị thổi phồng, **kể cả 87.72 của bài có giám sát** | **chưa kiểm** — repo chỉ tải được danh sách *tên taxa* của ToL-10M, không có ảnh |
| **1 seed** cho mọi run augmentation | chênh 1–3 điểm trên 113/20 lớp có thể là nhiễu. E1 đã bị chính điều này bác bỏ | chỉ CMO có kiểm chứng nội tại (liều–đáp ứng) |
| Mọi run CNN 30-epoch **thiếu train ~1.6 điểm** | con số tuyệt đối thấp hơn khả năng thật | đã đo, đã ghi cảnh báo |
| lr AdamW probe trên ConvNeXt rồi dùng cho Swin | chênh 1.77 giữa hai kiến trúc chưa phải kết luận sạch | đã ghi caveat |
| 167 mô tả T3b do LLM sinh, chưa ai soi | kết luận "mô tả riêng > chung" | 3.7 cho thấy nội dung mang thông tin thật |
| Caption CuPL sinh từ **tên lớp**, không phải từ ảnh | B4 là "text mức lớp có biến thiên" | đã ghi rõ |

---

## 6. Nhật ký các mốc

| Ngày | Mốc |
|---|---|
| 09-06 | EDA đầy đủ; chốt 3 hướng dự án |
| 09-06 | Sweep 8 run dự án 2 (~5 giờ) → 85.86% |
| 09-06 | Contamination 99.8% → **đổi trục** dự án 3 |
| 09-06 | Phát hiện bug QuickGELU → chạy lại toàn bộ phần CLIP |
| 09-06 | LoRA cho bài 555 lớp: 85.06 với 3.2% tham số |
| 09-07 | **Pass 3** CNN hiện đại: probe optimizer (SGD vs AdamW chênh **49 điểm**) → ConvNeXt-T IN22k 85.55 |
| 09-07 | **Pass 4** BioCLIP 87.72; **ViT@448 = 90.03 (kỷ lục)**; ensemble; bắt bug `inception transform_input` (−6.06) |
| 09-07 | **Pass 5** trục thời gian 2012–2022; bug `googlenet transform_input` (−7.11); thêm chốt chặn chống tái diễn |
| 09-07 | Logit adjustment hậu kiểm: **+5.78 F1 lớp khó, miễn phí**. Tải CUB-200-2011 bản đầy đủ |
| 09-07 | Khảo sát attention/encoder-decoder → [docs/05](docs/05-attention-encoder-decoder.md); viết lại mục E theo tài liệu |
| 09-07 | **Pass 6** augmentation 18 run: D3 xác nhận bằng số; CMO có liều–đáp ứng trên model yếu nhưng **hại model mạnh** |
| 09-07 | **Pass 8** part-query decoder: encoder đóng băng cho **+1.12 top-1, +4.23 recall lớp khó** |
| 09-07 | **Pass 7** NABirds → CUB: 3/3 model dương, TB +0.25. **Toàn bộ hàng đợi 8 pass hoàn tất** |
