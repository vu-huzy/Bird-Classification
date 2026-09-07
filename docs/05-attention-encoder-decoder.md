# 5. Hướng attention / encoder-decoder cho NABirds — khảo sát và đề xuất

> Tra cứu 2026-09-07. Mục tiêu: trả lời câu "có framework attention hoặc
> encoder-decoder nào áp được vào bài này không, và áp thì được gì".
>
> Kết luận ngắn: **có, nhưng phải chọn đúng biến thể.** Bản encoder-decoder kinh
> điển nhất (INTR) *giảm* accuracy chứ không tăng — nó đánh đổi lấy khả năng giải
> thích. Bản đáng làm cho repo này là **decoder với K "part query"**, và giá trị
> chính của nó không nằm ở accuracy mà ở một phép đo mà NABirds cho phép còn CUB
> thì đã bị làm mòn: **đối chiếu bản đồ attention với 11 keypoint bộ phận**.

---

## 1. Bốn nhóm phương pháp đã khảo sát

### 1.1 HERBS — SOTA hiện tại trên NABirds

*Fine-grained Visual Classification with High-temperature Refinement and
Background Suppression* (arXiv:2303.06442), hạng **#1 trên Papers with Code cho
cả CUB-200-2011 lẫn NABirds**, **vượt 93%** ở cả hai.

Hai module:
- **High-temperature refinement** — tinh chỉnh feature map ở nhiều thang, dùng
  nhiệt độ cao để ép model học đặc trưng đa dạng thay vì bám vào một tín hiệu.
- **Background suppression** — tách feature map thành foreground/background theo
  điểm tin cậy phân loại, rồi *dập* giá trị ở vùng tin cậy thấp.

**Vì sao chưa áp được ngay:** repo chỉ hỗ trợ backbone Swin, và bản đạt 93% dùng
Swin cỡ lớn ở độ phân giải cao — không vừa 12GB. README và config chi tiết của
họ không ghi rõ VRAM/epoch, nên phải tự đo trước khi cam kết.

**Nhưng ý tưởng *background suppression* thì rẻ và áp được**: repo đã đo được
bbox chim chỉ chiếm **median 28.3% diện tích ảnh** (docs/01), tức ~72% pixel là
nền. Đây là chỗ có dư địa rõ ràng.

### 1.2 INTR — đúng framework encoder-decoder, nhưng ĐỔI accuracy lấy interpretability

*A Simple Interpretable Transformer for Fine-Grained Image Classification and
Analysis* (ICLR 2024, arXiv:2311.04157). Đây chính xác là kiến trúc
encoder-decoder + cross-attention:

```
encoder (DETR-ResNet-50)  ->  đặc trưng theo patch
decoder                    <-  C query, MỖI LỚP MỘT QUERY
   self-attn giữa các query + cross-attn query -> patch
   -> vector đặc trưng riêng cho từng lớp
phân loại: argmax_c  wᵀ z_out^(c)      (w = "presence vector" dùng chung)
```

Trọng số cross-attention cho biết mỗi lớp nhìn vào vùng nào — đó là điểm bán của
bài báo.

**Con số phải đọc kỹ:**

| | INTR | ResNet-50 |
|---|---|---|
| CUB-200-2011 | **71.8%** | **83.8%** |
| Birds-525 | 97.4% | 98.5% |

Tác giả nói thẳng: *đạt accuracy cao không phải mục tiêu của bài báo; mục tiêu là
chứng minh khả năng giải thích.* Và hạn chế họ tự nêu: **C query phải đưa vào
decoder cùng lúc, nên tốn khi C lớn hơn số ô lưới N của feature map.**

→ Với NABirds thì **C = 555 > N = 196** (patch 16 ở ảnh 224). Rơi đúng vào vùng
xấu mà chính tác giả cảnh báo. **Không dùng nguyên bản.**

### 1.3 PDiscoFormer — bằng chứng part-based CÓ THỂ tăng accuracy

*PDiscoFormer: Relaxing Part Discovery Constraints with Vision Transformers*
(ECCV 2024, arXiv:2407.04538).

Không có decoder. Thay vào đó: **prototype bộ phận học được**, tính attention
bằng khoảng cách Euclid bình phương âm giữa đặc trưng patch và prototype, chuẩn
hoá bằng Gumbel-Softmax; embedding bộ phận = trung bình có trọng số.

Sáu loss: cross-entropy + total variation (ép vùng liền mạch) + entropy (mỗi
patch thuộc về một bộ phận) + orthogonality (các bộ phận không trùng nhau) +
presence (bộ phận nào cũng phải xuất hiện) + equivariance.

Backbone **DINOv2 ViT-B đóng băng** (chỉ mở class/position/register token).
**K = 8 bộ phận → CUB 88.79%**, đồng thời NMI 69.87 / ARI 43.49 cho chất lượng
phân vùng bộ phận.

→ Đây là phản ví dụ quan trọng cho INTR: **part-based cải thiện CẢ accuracy CẢ
khả năng giải thích**, miễn là không đặt một query cho mỗi lớp.

### 1.4 Saccadic Vision — hai giai đoạn ngoại vi → hố mắt

*Saccadic Vision for Fine-Grained Visual Classification* (arXiv:2509.15688).
**Một encoder duy nhất**, không phải encoder-decoder. Mã hoá ảnh thu nhỏ để sinh
priority map, rồi lấy mẫu các patch "điểm nhìn" ở độ phân giải cao tại các vị trí
ưu tiên và mã hoá lại. Attention có chọn lọc theo ngữ cảnh với hệ số toàn cục α
và trọng số per-fixation β.

Swin-B, ảnh 512 (patch ngoại vi và điểm nhìn đều 224), thường 4 điểm nhìn:
**NABirds 90.8%, CUB 91.8%.**

→ Đáng chú ý vì nó đạt gần SOTA mà **mỗi lần forward chỉ xử lý ảnh 224** — hợp
với ràng buộc 12GB hơn hẳn train thẳng ở 448/512.

### 1.5 Nhóm còn lại (ghi để khỏi tra lại)

PCT-ViT (Dual-path Semantic Perception + Dynamic Position Encoding +
Counterfactual Token Selection), Dual-Dependency Attention Transformer (tách
tương tác token thành nhánh phụ-thuộc-vị-trí và phụ-thuộc-ngữ-nghĩa, độ phức tạp
tuyến tính), Hierarchical Attention ViT. Đều là biến thể attention trên ViT, đều
báo cáo trên CUB + NABirds, đều cần train lại toàn bộ backbone.

---

## 2. Đề xuất cho repo này: decoder K "part query"

### 2.1 Kiến trúc

```
Encoder = backbone đã có (giữ nguyên, không viết lại)
    ConvNeXt-T IN22k / ViT-B/16 IN21k / BioCLIP
    -> token đặc trưng  [B, N, D]        (N = 196 ở 224, hoặc H'×W' với CNN)

Decoder = L lớp, K query bộ phận học được  [K, D]     (K = 8..16, KHÔNG phải 555)
    self-attn  giữa K query          -> các bộ phận "chia việc" cho nhau
    cross-attn K query -> N token    -> mỗi query hút về một vùng ảnh
    -> [B, K, D]

Head:  concat K embedding -> Linear(K·D, 555)
```

**Điểm khác INTR — và là lý do nó không dính hạn chế của INTR:** query gắn với
**bộ phận**, không gắn với **lớp**. K = 8..16 thay vì C = 555, nên K ≪ N thay vì
C > N. Chi phí decoder gần như không đáng kể so với backbone.

**Chống sập (lấy từ PDiscoFormer):** nếu không ràng buộc, mọi query sẽ hút về
cùng một vùng. Cần tối thiểu hai loss phụ trên bản đồ cross-attention:
orthogonality (các bản đồ ít chồng nhau) và presence (bộ phận nào cũng xuất
hiện). Không cần đủ 6 loss của PDiscoFormer ngay từ đầu.

### 2.2 Vì sao hướng này hợp riêng với repo này

1. **NABirds có 11 keypoint bộ phận**, và EDA đã đo mức hiện diện: mỏ 98.6%,
   đỉnh đầu 97.7%, mắt ~52%, trung bình 8.6/11 (docs/01). Trong khi đó tài liệu
   hiện nay hầu như luôn chấm điểm attention-vs-bộ-phận **trên CUB** (15 bộ
   phận). Làm phép đo đó **trên NABirds** — dataset lớn gấp ~4 lần CUB — là chỗ
   ít người đi và **repo đã có sẵn dữ liệu**.
2. **Nhắm đúng loại lỗi còn lại.** Model tốt nhất hiện tại sai **14.3% trong cùng
   loài** (bioclip) — tức nhầm giữa các biến thể giới tính/tuổi/bộ lông của CÙNG
   một loài. Khác biệt giữa các biến thể đó là **cục bộ theo bộ phận** (hoa văn
   đầu, vạch cánh, màu ức). Model dựa trên bộ phận nhắm thẳng vào đó.
3. **Nối được với dự án 3.** `zeroshot/src/variants.py` đã parse 555 lá thành
   (sex, age, season, morph, form). Nếu part query nào đó học được "hoa văn đầu"
   thì có thể kiểm tra trực tiếp: nó có tách được male/female của cùng loài
   không?

### 2.3 Kỳ vọng trung thực

**Đừng kỳ vọng nhảy vọt về accuracy.** Ba dữ kiện:

- PLAN đã ước lượng module FGVC cho **+1–2 điểm** so với baseline tương ứng.
- INTR cho thấy kiểu decoder có query **có thể tụt 12 điểm** nếu thiết kế sai.
- PDiscoFormer đạt CUB 88.79% nhưng với **DINOv2 đóng băng** — phần lớn công là
  của backbone tự giám sát, không phải của module bộ phận.

Nên đặt mục tiêu theo đúng thứ tự này:
1. **Chính** — một phép đo định lượng attention-vs-keypoint trên NABirds mà repo
   hiện chưa có gì tương đương. Kết quả âm cũng có giá trị.
2. **Phụ** — +1–2 điểm accuracy, và giảm riêng `err_same_species%`.

### 2.4 Nếu mục tiêu là ACCURACY chứ không phải interpretability

Thì đây không phải hướng đúng. Xếp theo tỉ lệ lợi/chi phí:

| việc | vì sao | chi phí |
|---|---|---|
| **Độ phân giải 384–448** cho backbone tốt nhất | đòn bẩy đã đo được +4.57 trên resnet50; ViT@448 đang chạy | cao |
| **Background suppression kiểu HERBS** | 72% pixel là nền (bbox median 28.3% diện tích); rẻ, không thêm backbone | trung bình |
| **Ensemble + TTA** | đã làm, không train gì | ~0 |
| Decoder part query | xem trên | trung bình |

---

## Nguồn

- Chou et al., *Fine-grained Visual Classification with High-temperature Refinement
  and Background Suppression*, arXiv:2303.06442 —
  <https://arxiv.org/abs/2303.06442>, code <https://github.com/chou141253/FGVC-HERBS>
- Paul et al., *A Simple Interpretable Transformer for Fine-Grained Image
  Classification and Analysis* (INTR), ICLR 2024, arXiv:2311.04157 —
  <https://arxiv.org/html/2311.04157v2>
- Aniraj et al., *PDiscoFormer: Relaxing Part Discovery Constraints with Vision
  Transformers*, ECCV 2024, arXiv:2407.04538 — <https://arxiv.org/html/2407.04538>
- *Saccadic Vision for Fine-Grained Visual Classification*, arXiv:2509.15688 —
  <https://arxiv.org/html/2509.15688>
- *PCT-ViT: A vision transformer incorporating fine-grained perception enhancement
  and counterfactual token selection*, Digital Signal Processing —
  <https://www.sciencedirect.com/science/article/abs/pii/S1051200425008127>
- *Dual-Dependency Attention Transformer for Fine-Grained Visual Classification* —
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC11014298/>
