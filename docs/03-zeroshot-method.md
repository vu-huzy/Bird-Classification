# 3. Dự án zero-shot (CLIP / METS) — phân tích tiền đề và phương pháp

> Tách từ README gốc. Đây là phần LẬP LUẬN: vì sao đổi trục zero-shot,
> vì sao phương pháp này là DeViSE chứ không phải CLIP, thiết kế prompt.

> **Ghi chú điều hướng:** các "mục N" trong file này trỏ tới chính file này
> (mục 1–10). Tham chiếu tới mục 11–18 nằm ở [04-zeroshot-results.md](04-zeroshot-results.md);
> "mục 0" là phần *Số liệu mới đo được* trong [02-supervised.md](02-supervised.md).

# Dự án #3 — Zero-shot kiểu CLIP / METS: phân tích tiền đề & kế hoạch (2026-09-06)

Phần này viết **trước khi code**, hiệu chỉnh mục "4. Hướng METS" ở trên. Ba phát hiện mới
(đo trên chính dataset + kiểm tra HF API) làm thay đổi thiết kế thí nghiệm.

## 1. Ba phát hiện làm đổi kế hoạch

### 1.1 NABirds KHÔNG có tên khoa học — prompt gốc của BioCLIP không dựng được từ file có sẵn

`classes.txt` và `hierarchy.txt` chỉ chứa **tên tiếng Anh thông thường**:

| Tầng | NABirds ghi | BioCLIP cần |
|---|---|---|
| order (22 node depth-1) | `Perching Birds`, `Ducks, Geese, and Swans` | `Passeriformes`, `Anseriformes` |
| species (404 node) | `Yellow-rumped Warbler` | `Setophaga coronata` |
| family | *không có tầng family nhất quán* (xem mục "1011 node") | `Parulidae` |

`grep -icE "scientific|latin|genus" nabirds/README` -> **0**. Không có tên Latin ở bất kỳ đâu trong bộ dữ liệu.

BioCLIP được train với **hỗn hợp** kiểu text (taxonomic string Latin / scientific name / common name /
taxonomic+common) nên prompt chỉ-common-name vẫn chạy được. Nhưng biến thể mạnh nhất — và biến thể
cần cho bước kiểm tra contamination — đòi hỏi bảng ánh xạ ngoài:

> **Việc bắt buộc làm đầu tiên:** 404 common name -> `(order, family, genus, species)` Latin.
> Nguồn: Clements/eBird checklist (NABirds do Cornell làm, common name kỳ vọng khớp gần 1-1) hoặc
> GBIF vernacular-name API. **Chưa đo tỉ lệ khớp** — NABirds là bản 2015, taxonomy chim đã có
> tách/gộp loài từ đó.

### 1.2 Prompt chỉ-tên-loài bị chặn trần **78.75%** ở bài 555 lớp

Đo trên đúng test set (24,633 ảnh):

| Đại lượng | Giá trị |
|---|---|
| 555 lá gộp về species (cha trực tiếp) | 404 species |
| Species có đúng 1 lá | 267 |
| Species có 2 / 3 / 4 / 5 lá | 126 / 9 / 1 / 1 -> **137 species, 288 lá** |
| Ảnh test thuộc species đa-lá | **11,955 / 24,633 = 48.5%** |
| **Trần top-1 555-way nếu text chỉ mang tên loài** | **78.75%** |

Trần này = giả định đoán đúng species 100% rồi luôn chọn lá đông nhất trong species đó. Nó **thấp hơn**
baseline có giám sát đã đo (`vit_b_16_in21k` = 85.86%). Hệ quả:

1. Con số zero-shot **chính** phải báo cáo ở mức **404 species**, không phải 555 lá.
2. Muốn đánh 555 lá thì text **bắt buộc** phải mang thông tin giới tính / tuổi / bộ lông.
   Phần "descriptor sinh bằng LLM" (CuPL/DCLIP) do đó **không phải tùy chọn làm đẹp** mà là
   thành phần cốt lõi của phương pháp.

### 1.3 BioCLIP gần như chắc chắn đã thấy 404 loài này -> "zero-shot mức loài" không dùng được

TreeOfLife-10M = **iNat21 + EOL + BIOSCAN-1M**. EOL phủ gần như mọi loài đã được mô tả; 404 loài chim
Bắc Mỹ đều phổ biến và nhiều ảnh. -> Xác suất trùng gần 100%. *(Đây là suy luận từ cấu tạo nguồn dữ
liệu, **chưa phải con số đo**.)*

Cách đo chính xác — đã kiểm tra file tồn tại qua HF API, `huggingface.co` truy cập được (HTTP 200):

| File trên dataset `imageomics/TreeOfLife-10M` | Kích thước | Dùng để |
|---|---|---|
| `embeddings/txt_emb_species.json` | 66 MB | Danh sách species BioCLIP đã embed — **rẻ nhất, dùng cái này** |
| `metadata/species_level_taxonomy_chains.csv` | 303 MB | Chuỗi taxonomy đầy đủ, đối chiếu cả mức genus/family |
| `metadata/catalog.csv` | 2.0 GB | Không cần |

Bước này **phụ thuộc vào 1.1** (đối chiếu phải bằng tên Latin).

## 2. Đề xuất: đổi trục "zero-shot" từ *loài* sang *biến thể bộ lông*

> **Thuật ngữ (D11).** Trục biến thể là **compositional zero-shot**: model đã thấy
> ẢNH chim mái khi pretrain, chỉ chưa thấy NHÃN "female". Không phải ZSL cổ điển.
> Mọi chỗ viết "zero-shot" cho trục biến thể trong file này đều hiểu theo nghĩa này.

Vì trục loài đã nhiễm, trục còn lại sạch và **độc nhất ở NABirds**:

> BioCLIP được giám sát **hoàn toàn bằng taxonomy**. Nhãn "Adult male" / "Female/immature" /
> "Breeding Myrtle" **không tồn tại** trong nguồn train của nó. Nên dự đoán 555 lá là bài toán
> nhãn-mới thật sự, ngay cả với loài mà model đã thấy.

Số liệu hỗ trợ (đo được):

| | Giá trị |
|---|---|
| Lá là biến thể (có ngoặc đơn) | 288 / 555 |
| Số **chuỗi biến thể khác nhau** (đã lower+strip) | **52** |
| Trục ngữ nghĩa trong 52 chuỗi | giới tính (male/female), tuổi (adult/immature/juvenile), mùa (breeding/nonbreeding/winter/summer/eclipse), morph màu (dark/light/white/blue/red/sooty/slate-colored/tan-striped/pink-sided), nhóm phụ loài (Myrtle/Audubon, Oregon, red-shafted/yellow-shafted, thick-billed) |
| Dữ liệu bẩn cần chuẩn hoá | `Adult Male` vs `Adult male`; `Female/immature male` vs `Female/Immature male`; `Adult ` (thừa dấu cách) |

**Split unseen đề xuất (đã tính):** giữ lại các lá "female/immature" của những species mà lá "male"
vẫn nằm trong tập train.

| | Giá trị |
|---|---|
| Species có **cả** lá male và lá female/immature | **81** |
| Lá unseen (female/immature) | 81 |
| **Ảnh test unseen** | **2,967** |
| Ảnh train của các lá male tương ứng | 3,816 |

Câu hỏi nghiên cứu: *"Text encoder cần biết gì để tách được biến thể bộ lông mà taxonomy không mã hoá?"*

## 3. Hai điểm kỹ thuật phải sửa so với công thức CLIP/METS mặc định

### 3.1 Text đóng băng + text mức-lớp => **không cần batch lớn, không cần gradient accumulation**

Chỉ có tối đa 555 chuỗi text duy nhất. Precompute một lần, mỗi step so ảnh với **toàn bộ 555 prototype**:

```
logits = (img_emb @ text_proto.T) / tau      # (B, 555)
loss   = CrossEntropy(logits, y)
```

Số negative = 555 **bất kể batch size**. Đây là lý do toàn bộ hướng này rẻ: chi phí đúng bằng chi phí
train phân loại thường đã đo (`vit_b_16_in21k` @224, 25 epoch = **28 phút** trên RTX 5070).

### 3.2 Phải gọi đúng tên phương pháp: đây là **ZSL dựa trên semantic embedding**, không phải contrastive kiểu CLIP

| | CLIP | METS (MIDL'23) | NABirds |
|---|---|---|---|
| Độ hạt của text | **mỗi ảnh 1 alt-text** | **mỗi ECG 1 báo cáo** | **mỗi LỚP 1 mô tả** |

CLIP và METS đều có text **mức instance**. NABirds không có caption theo ảnh (xem mục 4 phía trên) nên
tốt nhất chỉ đạt text **mức lớp**. Hệ quả:

- Nếu vẫn dùng InfoNCE in-batch mà không mask, các ảnh **cùng lớp** trong batch trở thành false
  negative — model bị phạt vì kéo hai ảnh cùng loài lại gần nhau.
- Cách xử lý: (a) mask/gộp same-class kiểu SupCon, hoặc (b) **tốt hơn** — bỏ hẳn in-batch negative,
  dùng công thức 3.1.
- Với công thức (b), phương pháp này **về mặt toán học là cross-entropy với classifier head cố định
  bằng embedding text** — tức dòng DeViSE / ALE / ESZSL, không phải CLIP. Ghi rõ trong báo cáo,
  nếu không sẽ bị phản biện đúng điểm này.

Đây không phải điểm yếu: nó vẫn trả lời được câu hỏi của METS ("frozen LM có giúp tổng quát hoá sang
lớp chưa thấy không"), chỉ là phải gọi đúng tên.

## 4. Kế hoạch thực thi

Môi trường đã có: torch 2.12.0.dev+cu128, timm 1.0.29, transformers 4.51.3, sentence-transformers 4.1.0,
RTX 5070 12.8GB, 483 GB trống. **Thiếu: `open_clip_torch`** (`pip install open_clip_torch`).

Checkpoint (đã xác minh qua `open_clip_config.json` trên HF):

| Model | Vision | embed_dim | Ghi chú |
|---|---|---|---|
| `openai/clip-vit-base-patch16` | ViT-B/16 | 512 | mốc domain-generic bắt buộc |
| `imageomics/bioclip` | ViT-B/16, 224 | 512 | domain-matched, cùng cỡ CLIP -> so sánh sạch |
| `imageomics/bioclip-2` | **ViT-L/14**, 24 layer, width 1024 | **768** | ToL-200M. Inference chắc chắn vừa 12GB; full fine-tune thì căng -> chỉ dùng cho bước 2, hoặc LoRA |

Cả 3 dùng chuẩn hoá CLIP (`mean=[0.4815, 0.4578, 0.4082]`), **không phải** ImageNet norm mà pipeline
hiện tại đang dùng — phải tách transform riêng.

| # | Bước | Verify | Chi phí |
|---|---|---|---|
| 0 | `pip install open_clip_torch`; ánh xạ 404 common name -> taxonomy Latin; parser 52 chuỗi biến thể -> `(sex, age, season, morph, subspecies)` | >=95% loài khớp tự động, in danh sách miss; **555/555** lá parse được, 0 lá rơi vào `unknown` | 1 buổi |
| 1 | Contamination audit: tải `txt_emb_species.json`, đối chiếu 404 tên Latin | Ra được con số `X/404 loài NABirds thuộc ToL-10M` -> quyết định gọi là "zero-shot" hay "đánh giá in-domain" | 2–3 giờ |
| 2 | **Zero-shot off-the-shelf, không train gì.** Ma trận {CLIP, BioCLIP, BioCLIP-2} x {T0..T3} | BioCLIP > CLIP ở mức 404-way (nếu không -> bug ở prompt hoặc normalization); T0/T1 ở 555-way phải **<= 78.75%** | 1 buổi |
| 3 | **METS-analogue.** Fine-tune image tower, text đóng băng, train trên seen, test 81 lá unseen. 3 text tower x 2 init image tower | So với bước 2 trên **cùng** 81 lá unseen; báo cáo GZSL harmonic mean | ~30 phút/run |
| 4 | Báo cáo: seen / unseen / harmonic mean x {555 lá, 404 species, 22 order} | — | — |

Bốn mức text cho bước 2 (biến duy nhất, image tower giữ nguyên):

| | Nội dung prompt |
|---|---|
| **T0** | `a photo of a {common name}` — mốc CLIP thuần |
| **T1** | taxonomy Latin đầy đủ + common name — **format gốc của BioCLIP** |
| **T2** | T1 + cụm biến thể đã chuẩn hoá (`adult male`, `female or immature`) |
| **T3** | T2 + 5–8 câu mô tả hình thái sinh bởi LLM (CuPL), lấy trung bình embedding |

> **Mẹo làm cho bước 2 gần như miễn phí:** embed 24,633 ảnh test **một lần** cho mỗi image tower rồi
> cache ra `.npy`. Sau đó mọi biến thể prompt chỉ là một phép nhân ma trận `(24633, D) @ (D, 555)` —
> thử bao nhiêu prompt cũng được, tính bằng giây.

## 5. Rủi ro / điều CHƯA xác minh

| Rủi ro | Trạng thái |
|---|---|
| Tỉ lệ khớp common name NABirds(2015) với eBird/Clements hiện tại | **chưa đo** — có tách/gộp loài trong 10 năm |
| License và cách tải tự động Clements checklist | **chưa kiểm** |
| Mô tả hình thái do LLM sinh có thể **bịa** | phải kiểm chéo tay ~30 lớp trước khi dùng cho T3 |
| BioCLIP-2 ViT-L/14 full fine-tune trên 12GB | nhiều khả năng **không đủ** — dự phòng LoRA/adapter, hoặc chỉ dùng inference |
| Image tower init từ BioCLIP đã thấy ảnh chim trên web | không tránh được -> chạy song song init từ `vit_b_16_in21k` (đã có checkpoint 85.86%) làm đối chứng sạch hơn |

## 6. Việc bị loại khỏi kế hoạch (và lý do)

- **Train contrastive trên CUB rồi test NABirds** (kế hoạch cũ ở mục 4): vẫn hợp lệ nhưng 6k ảnh CUB
  quá nhỏ so với BioCLIP, và trục loài đã nhiễm sẵn nên không thu được kết luận sạch hơn trục biến thể.
  Hạ xuống ưu tiên thấp.
- **Batch lớn / gradient accumulation cho contrastive**: không cần, xem mục 3.1.
- **Split unseen theo order hiếm** (Parrots/Nightjars/Storks, 236 ảnh): quá nhỏ để có ý nghĩa thống kê.

## 7. Làm rõ: **hai trục độc lập**, đừng gộp

Câu hỏi "nguồn text là gì" và "train cái gì trên dữ liệu nào" là **hai trục vuông góc**. Có thể
chọn tự do một mức ở trục A và một mức ở trục B.

### 7.1 Trục A — chuỗi text đưa vào text encoder (đóng băng)

BioCLIP **không có caption người viết**: nó lấy **chính chuỗi nhãn phân loại làm text**. Đó là mức T1.
Ví dụ thật, species `Yellow-rumped Warbler` có **4 lá**:

```
Yellow-rumped Warbler (Breeding Myrtle)
Yellow-rumped Warbler (Winter/juvenile Myrtle)
Yellow-rumped Warbler (Breeding Audubon's)
Yellow-rumped Warbler (Winter/juvenile Audubon's)
```

| Mức | Chuỗi text thực tế sinh ra | Tách được 4 lá? |
|---|---|---|
| **T0** | `a photo of a Yellow-rumped Warbler` | **Không** — 4 lá cho ra cùng 1 chuỗi |
| **T1** (format gốc BioCLIP) | `Animalia Chordata Aves Passeriformes Parulidae Setophaga coronata` | **Không** — taxonomy dừng ở mức loài |
| **T2** | `... Setophaga coronata with common name Yellow-rumped Warbler, breeding Myrtle form` | **Có** — 4 chuỗi khác nhau |
| **T3** | T2 + `"...mảng vàng ở hông và hai bên sườn, cổ họng TRẮNG (Myrtle), hai vạch cánh trắng..."` | Có, và giàu tín hiệu thị giác hơn |

**Điểm mấu chốt:** T2 chính là *"dùng nhãn làm mô tả"* — chỉ khác ở chỗ dùng **tên lá đầy đủ** của
NABirds thay vì cắt về tên loài. **Không cần LLM.** Đây là mức tối thiểu để vượt trần 78.75% (mục 1.2).

T3 chỉ trả lời một câu hỏi ablation cụ thể: *nhãn lá đã đủ chưa, hay text còn cần kiến thức hình thái?*
`(Breeding Myrtle)` và `(Breeding Audubon's)` là hai chuỗi khác nhau, nhưng text encoder **không biết**
Myrtle khác Audubon ở chỗ nào (cổ họng trắng vs vàng). LLM/Wikipedia bù đúng khoảng trống đó.

> **Chiến lược giảm rủi ro (khuyến nghị):** chạy **T0 -> T2 trước**, xem T2 có vượt 78.75% không.
> - Vượt rõ -> T3 chỉ là phần thưởng thêm.
> - T2 kẹt -> lúc đó mới **biết chắc** là thiếu kiến thức hình thái, và T3 có lý do rõ ràng để làm.
>
> Cách này biến T3 từ "công việc đoán mò" thành "phản hồi cho một kết quả đã đo".

### 7.2 Trục B — train cái gì, trên dữ liệu nào

| | Text tower | Image tower | Dữ liệu train | METS-faithful? |
|---|---|---|---|---|
| **B0** off-the-shelf | đóng băng | đóng băng | không train | — (mốc tham chiếu) |
| **B1** output layer | đóng băng | đóng băng + linear proj **train** | NABirds seen | gần |
| **B2** LoRA | đóng băng | **LoRA** trên image tower | NABirds seen | gần |
| **B3** full fine-tune | đóng băng | **toàn bộ** | NABirds seen | gần |
| **B4** CUB caption | đóng băng | train được | **CUB (ảnh, caption)** | **đúng nhất** |

### 7.3 Vai trò của CUB — đã sửa so với mục 6

Mục 6 hạ CUB xuống ưu tiên thấp vì **pretrain ảnh** trên CUB là vô ích: CUB có 5,994 ảnh train,
NABirds có 23,929 (gấp 4x), và 71% loài CUB đã nằm trong NABirds. Kết luận đó vẫn đúng **cho mục đích
pretrain ảnh**.

**Nhưng CUB có một thứ NABirds không có và không thể tự tạo:** caption **theo từng ảnh**
(Reed et al. 2016 — 10 câu/ảnh, ~118k caption) và 312 attribute nhị phân.

Đó chính xác là thứ mục 3.2 chỉ ra là đang thiếu: **text mức instance**. Không có nó thì phương pháp
là DeViSE/ALE; có nó thì mới đúng là contrastive kiểu CLIP/METS (METS ghép mỗi ECG với **một** báo cáo
riêng của ca đó).

> **Vai trò đúng của CUB: không phải để pretrain ảnh, mà để train alignment bằng caption mức instance
> — đúng cấu trúc (ECG, báo cáo) của METS — rồi zero-shot sang NABirds.** Đây là biến thể B4, trung
> thành với METS nhất trong toàn bộ kế hoạch.

Cảnh báo trung thực bắt buộc ghi vào báo cáo: B4 train trên 6k ảnh, **gần như chắc chắn thua BioCLIP**
đã học 10M ảnh. Framing đúng là *"alignment nhỏ và chuyên biệt có tổng quát hoá sang lớp chưa thấy
không"*, **không phải** *"tôi đánh bại BioCLIP"*.

**Trạng thái:** CUB **chưa có trên máy** (đã tìm, không thấy). Cần tải `CUB_200_2011.tgz` (~1.1 GB)
+ bộ caption của Reed et al. (~100 MB). Chỉ cần khi làm B4.

## 8. Quyết định đã chốt

| Quyết định | Chọn | Hệ quả |
|---|---|---|
| **Trục zero-shot** | **Cả hai, báo cáo song song** | (a) Biến thể bộ lông: 81 lá female/immature unseen, 2,967 ảnh test. (b) Holdout loài: ~20% species, stratify theo order. Bước 3 chạy 2 lần -> thêm ~1 giờ |
| **Nguồn text** | **T0 -> T2 trước, T3 có điều kiện** | Vòng đầu **không cần LLM sinh mô tả**. Chỉ làm T3 nếu T2 không vượt trần 78.75% |
| **BioCLIP-2** | **Có, và thử LoRA** | Thêm `pip install peft`. Xem gotcha ở mục 9 |

Ưu tiên trục B: **B0 (bước 2) -> B1 -> B2/LoRA -> B4 (CUB)**. B3 full fine-tune bỏ qua với ViT-L/14.

## 9. Gotcha kỹ thuật cho LoRA trên open_clip — **ĐÃ XÁC MINH** (kết quả ở mục 10.4)

`peft` **chưa cài** (`accelerate 1.6.0`, `safetensors 0.5.3`, `huggingface-hub 0.30.2` đã có).

Vấn đề, **đã xác minh trên open_clip 3.3.0**: khối attention của open_clip dùng
`nn.MultiheadAttention`, trong đó q/k/v gộp thành tham số `in_proj_weight` — **không phải `nn.Linear`**.
`peft` chỉ gắn LoRA được vào `nn.Linear`, nên `target_modules` sẽ **không bắt được q/k/v**.

Kiểm tra 1 dòng sau khi cài:

```python
import open_clip
m, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
print({type(mod).__name__ for n, mod in m.visual.named_modules() if 'attn' in n})
```

Ba đường thoát, theo thứ tự ưu tiên:

1. **LoRA chỉ trên MLP + `out_proj`** (`c_fc`, `c_proj`, `out_proj` đều là `nn.Linear`) — chạy được
   ngay, yếu hơn một chút vì không chạm q/k/v.
2. **Load cùng trọng số qua `timm`** — ViT của timm có `qkv` gộp thành **một `nn.Linear`**, `peft`
   bắt được. Đổi lại phải tự xử lý projection head và tokenizer.
3. Tự viết wrapper LoRA cho `MultiheadAttention` — tốn công nhất, chỉ làm nếu (1) và (2) đều hỏng.

**Verify cho bước LoRA:** VRAM peak < 12 GB ở batch 32, bf16, 224px; và LoRA phải **thắng B1
(linear-proj-only)** trên cùng tập unseen — nếu không thì LoRA không đáng chi phí, quay về B1.

## 10. Bước 0 — ĐÃ CHẠY XONG (2026-09-06)

Toàn bộ thí nghiệm zero-shot nằm trong `zeroshot/`, **tách khỏi** `src/` của bài 555 lớp.

```
zeroshot/
  src/zs_env.py           thiết lập môi trường, phải import ĐẦU TIÊN
  src/variants.py         parser 60 chuỗi biến thể -> (sex, age, season, morph, form, phrase)
  src/build_variants.py   -> data/leaf_variants.csv    (555 dòng)
  src/build_taxonomy.py   -> data/species_taxonomy.csv (404 dòng, qua GBIF)
  data/                   2 file CSV trên (124 KB, commit được)
  cache/gbif/             404 JSON GBIF (18 MB, đã cho vào .gitignore)
  results/                (trống, dành cho bước 2+)
```

```bash
python zeroshot/src/build_variants.py     # 555 lá  -> leaf_variants.csv
python zeroshot/src/build_taxonomy.py     # 404 loài -> species_taxonomy.csv (cache -> chạy lại tức thì)
```

### 10.1 Kết quả verify

| Kiểm tra | Yêu cầu | Đạt |
|---|---|---|
| Lá parse được | 555 | **555** |
| Lá có biến thể | 288 | **288** |
| Chuỗi biến thể khác nhau | 60 | **60** |
| Lá **không** parse được (`known=False`) | 0 | **0** |
| `base_name` của lá == tên species cha | 0 lệch | **0 lệch** |
| Loài khớp tên thường GBIF chính xác | >=95% | **404/404 = 100%** |
| Tên nhị thức hợp lệ | 404 | **404** |
| Tên khoa học duy nhất (không đụng độ) | 404 | **404** |
| JOIN 555 lá + taxonomy | 0 thiếu | **0 thiếu** |

Cài thêm: `open_clip_torch 3.3.0`, `peft 0.20.0`, `ftfy 6.3.1`.

### 10.2 Hai bug đã gặp khi tra GBIF (ghi lại để khỏi mất thời gian lần sau)

**`Redhead` trả về toàn nấm.** Tra tên thường một-từ phổ thông thì GBIF full-text
search trúng `Agaricomycetes` (nấm) chứ không trúng con vịt. -> `ALIAS = {'Redhead':
'Aythya americana'}`. Quan trọng: alias chỉ đổi **chuỗi đem đi tra**, việc xác minh
vẫn so tên NABirds gốc với danh sách vernacular của GBIF, nên không nới lỏng tiêu chuẩn.

**`American Black Duck` khớp trúng bản ghi LAI.** GBIF trả `Anas rubripes x
platyrhynchos` (Black Duck x Mallard) — bản ghi này **cũng** mang vernacular
"American Black Duck", nên qua được vòng kiểm tra, nhưng bản ghi lai **không có
`canonicalName`** -> cột `scientific_name` ra rỗng trong khi vẫn báo là khớp.
Sửa: bắt buộc `canonicalName` phải là **tên nhị thức 2 từ**. Đáp án đúng: `Anas rubripes`.

> Bài học chung: "khớp tên thường" **không đủ** để nói là khớp đúng. Ba kiểm tra bổ
> sung mới bắt được lỗi: tên nhị thức hợp lệ, tên khoa học duy nhất, và JOIN không thiếu dòng.

### 10.3 22 "order" tiếng Anh của NABirds -> 20 order Latin

Ánh xạ **sạch**: mỗi nhóm tiếng Anh rơi trọn vào **đúng một** order Latin, không nhóm nào bị xẻ.
Con số giảm từ 22 xuống 20 chỉ vì Charadriiformes gộp ba nhóm của NABirds:

| Nhóm NABirds | Order Latin |
|---|---|
| `Gulls, Terns, and Allies` (17 loài) | Charadriiformes |
| `Plovers, Sandpipers, and Allies` (27) | Charadriiformes |
| `Skuas and Alcids` (2) | Charadriiformes |
| `Perching Birds` (205) | Passeriformes |
| 18 nhóm còn lại | 1-1 với order Latin tương ứng |

-> Có thể báo cáo accuracy ở **cả hai** mức: 22 nhóm NABirds (so được với bảng kết quả
có giám sát đã có) và 20 order Latin (so được với BioCLIP).

### 10.4 LoRA trên open_clip — gotcha ở mục 9 đã xác minh là ĐÚNG

Chạy trên `open_clip 3.3.0`, cả `ViT-B-16` lẫn `ViT-L-14`:

```
transformer.resblocks.N.attn           -> nn.MultiheadAttention   (q/k/v gộp trong in_proj_weight)
transformer.resblocks.N.attn.out_proj  -> NonDynamicallyQuantizableLinear  (là nn.Linear)
transformer.resblocks.N.mlp.c_fc       -> nn.Linear
transformer.resblocks.N.mlp.c_proj     -> nn.Linear
```

`nn.Linear` duy nhất trong `visual`: **`c_fc`, `c_proj`, `out_proj`**. q/k/v nằm trong
`in_proj_weight` là `nn.Parameter` thô -> **`peft` không gắn LoRA vào q/k/v được**.

-> Chốt phương án 1 ở mục 9: `target_modules=['c_fc','c_proj','out_proj']`. Nếu kết quả
không đủ, chuyển sang phương án 2 (load qua `timm`, có `qkv` gộp thành một `nn.Linear`).

### 10.5 Gotcha môi trường: `import peft` làm chết Python

Máy này có TensorFlow build theo NumPy 1.x, NumPy hiện tại là 2.4.6.
`import peft` -> `transformers.image_transforms` -> `import tensorflow` -> `ml_dtypes`
-> `ImportError: numpy.core.umath failed to import`.

Sửa **không đụng vào môi trường**: `zs_env.py` set `USE_TF=0` + `TRANSFORMERS_NO_TF=1`
trước mọi import nặng. Vì vậy **mọi script trong `zeroshot/` phải `import zs_env` đầu tiên**.
(`import open_clip` một mình thì không sao — chỉ `peft`/`transformers` mới dính.)

### 10.6 Hai file dữ liệu sinh ra

**`zeroshot/data/leaf_variants.csv`** — 555 dòng:
`class_id, leaf_name, base_name, species_name, order_name, variant_raw, phrase,
sex, age, season, morph, form, female_side, male_only, n_train, n_test`

**`zeroshot/data/species_taxonomy.csv`** — 404 dòng:
`species_class_id, common_name, query_used, scientific_name, kingdom, phylum, class,
order, family, genus, match`

JOIN theo `species_name` = `common_name` -> 555 dòng đủ thông tin dựng prompt:

```
Yellow-rumped Warbler (Breeding Myrtle)
  T0: a photo of a Yellow-rumped Warbler.
  T1: a photo of Animalia Chordata Aves Passeriformes Parulidae Setophaga coronata
      with common name Yellow-rumped Warbler.
  T2: ... with common name Yellow-rumped Warbler, breeding Myrtle form.

Baltimore Oriole (Female/Immature male)
  T2: ... Icteridae Icterus galbula with common name Baltimore Oriole, female or immature male.

Snow Goose (Blue morph)
  T2: ... Anatidae Anser caerulescens with common name Snow Goose, blue morph.
```

### 10.7 Split unseen "biến thể" — số chốt

| | Giá trị |
|---|---|
| Species có cả lá male-only và lá female-side | **81** |
| Lá unseen (female/immature) | **81** |
| Ảnh test unseen | **2,967** |
| Lá male tương ứng (seen), ảnh train | **86 lá** (vài loài có 2 lá male, vd `Long-tailed Duck` Winter/Summer), **3,816 ảnh** |
| Ảnh train bị bỏ khi giữ 81 lá làm unseen | 2,869 |

Đã soi tay: mọi cặp đều hợp lý — `Baltimore Oriole (Adult male)` [seen] /
`(Female/Immature male)` [unseen], `American Goldfinch (Breeding Male)` /
`(Female/Nonbreeding Male)`, `American Kestrel (Adult male)` / `(Female, immature)`...

Một ca biên: `Common Eider (Immature/Eclipse male)` bị xếp `male_only`. Đúng về mặt
bộ lông (không phải lớp "mái"), nhưng "Immature" về lý thuyết có thể gồm cả con mái.
Chỉ ảnh hưởng 1/81 lá — ghi lại để nếu cần thì loại ra khi làm ablation.

### 10.8 Việc tiếp theo

Bước 1 — contamination audit: tải `embeddings/txt_emb_species.json` (66 MB) từ HF
dataset `imageomics/TreeOfLife-10M`, đối chiếu với 404 tên Latin vừa dựng, ra con số
`X/404 loài NABirds thuộc ToL-10M`.

---
