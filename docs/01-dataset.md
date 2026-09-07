# 1. Dataset NABirds — cấu trúc, taxonomy, EDA

> Tách từ README gốc. Mọi số liệu ở đây đo trực tiếp trên dataset.

## Cấu trúc dữ liệu (`nabirds/`)

Bài toán gốc: **555 lớp lá** (species/giới tính/tuổi/bộ lông, ví dụ *"Baltimore Oriole (Female/Immature male)"*
được tính là 1 lớp riêng), tổ chức theo cây phân loại (order → family/genus → species).

| File | Vai trò |
|---|---|
| `images/<555 thư mục>/*.jpg` | 48,562 ảnh gốc, 1 thư mục = 1 lớp lá |
| `images.txt` | `<image_id> <đường dẫn ảnh>` |
| `image_class_labels.txt` | `<image_id> <class_id>` — nhãn ground-truth |
| `classes.txt` | `<class_id> <tên lớp>` — 1011 dòng (555 lá + 456 node cha order/family/genus) |
| `hierarchy.txt` | `<child_class_id> <parent_class_id>` — cây phân loại, root = `class_id 0` |
| `bounding_boxes.txt` | `<image_id> <x> <y> <w> <h>` — 1 bbox/ảnh bao quanh con chim |
| `sizes.txt` | `<image_id> <width> <height>` — kích thước gốc, dùng thay vì mở ảnh để đo |
| `train_test_split.txt` | `<image_id> <is_train>` — split gợi ý (1=train, 0=test) |
| `photographers.txt` | `<image_id> <tên nhiếp ảnh gia>` — metadata, **không phải feature** |
| `parts/parts.txt` | tên 11 bộ phận cơ thể (mỏ, đầu, mắt trái/phải, bụng, ngực, lưng, đuôi, cánh trái/phải) |
| `parts/part_locs.txt` | `<image_id> <part_id> <x> <y> <visible>` — keypoint từng ảnh |
| `nabirds.py`, `__init__.py`, `README`/`README~` | script mẫu (Python 2, chỉ tham khảo) + tài liệu gốc |

## Cách hiểu taxonomy, kích thước ảnh và annotation

### 1011 node không có nghĩa là 1010 lớp con thuộc 555 lớp cha

`classes.txt` có 1011 node trong cây phân loại, gồm:

- **555 lớp lá**: lớp cuối cùng dùng làm nhãn phân loại ảnh. Mỗi ảnh trong
  `image_class_labels.txt` có đúng một `class_id` thuộc nhóm này.
- **456 node cha**: các nhóm trung gian như order, family hoặc genus. Chúng
  không phải là một bộ 555 nhãn khác và không được gán trực tiếp cho ảnh trong
  bài toán 555 lớp.

Quan hệ cha-con được lưu trong `hierarchy.txt`. Ví dụ, từ một lớp lá ta lần
theo `child_class_id -> parent_class_id` để biết ảnh thuộc những nhóm lớn nào.
Vì vậy cần phân biệt:

```text
ảnh -> class_id lớp lá -> các class_id node cha qua hierarchy.txt
```

Thư mục trong `images/` cũng được tổ chức theo 555 lớp lá. Không dùng số thứ tự
dòng để nối các file; luôn nối bằng `image_id` hoặc `class_id`.

### Mỗi ảnh có kích thước gốc khác nhau

Đúng. `sizes.txt` ghi kích thước gốc của từng ảnh, nhưng CNN cần một kích thước
đầu vào thống nhất trong mỗi batch. Resize không làm mất nhãn; nó chỉ thay đổi
ảnh đầu vào. Không cần tìm một kích thước duy nhất có thể giữ nguyên toàn bộ
chi tiết của mọi ảnh, vì mọi mô hình CNN đều phải đánh đổi giữa chi tiết và chi
phí tính toán.

Khuyến nghị theo thứ tự thử nghiệm:

1. **Baseline:** resize ảnh sao cho cạnh ngắn là `256`, sau đó crop ngẫu nhiên
   `224x224` khi train và center crop `224x224` khi validation/test. Đây là cấu
   hình nhẹ, phổ biến và phù hợp để kiểm tra pipeline.
2. **Khi cần giữ chim đầy đủ hơn:** dùng resize giữ aspect ratio rồi padding
   (letterbox) vào `224x224`, thay vì crop. Cách này không cắt chim nhưng có thể
   thêm vùng nền.
3. **Khi GPU đủ và cần nhiều chi tiết:** thử `320x320` hoặc `384x384` sau khi
   baseline chạy ổn. Không nên bắt đầu ở độ phân giải cao vì tốn VRAM và thời
   gian hơn mà chưa biết accuracy có tăng hay không.

Với dữ liệu này, nên bắt đầu bằng **ảnh đầy đủ, `224x224`, không crop bbox**.
EDA cho thấy bbox trung bình chỉ chiếm khoảng 30.7% diện tích ảnh, nên crop bắt
buộc có thể làm mất bối cảnh hữu ích. Sau đó có thể chạy một thí nghiệm đối
chứng với ảnh crop theo bbox.

### Dùng bounding box như thế nào?

`bounding_boxes.txt` có dạng:

```text
image_id x y width height
```

`x, y` là góc trên bên trái; `width, height` là kích thước khung theo pixel
trên ảnh gốc. Với dòng:

```text
<image_id> 83 59 128 228
```

vùng cắt là `(left=83, top=59, right=211, bottom=287)`. Khi crop nên kẹp
tọa độ vào biên ảnh vì có 201 annotation vượt biên nhẹ:

```text
left   = max(0, x)
top    = max(0, y)
right  = min(image_width, x + width)
bottom = min(image_height, y + height)
```

Có hai cách dùng chính:

- **Ảnh đầy đủ:** không cần biến đổi bbox; chỉ dùng bbox để kiểm tra, vẽ hoặc
  thử nghiệm phụ.
- **Ảnh crop chim:** crop vùng bbox trước, sau đó resize crop về `224x224`.
  Khi đó nhãn lớp vẫn giữ nguyên, nhưng model không còn nhìn thấy phần lớn bối
  cảnh. Nên so sánh kết quả với baseline ảnh đầy đủ.

Nếu resize ảnh mà vẫn muốn vẽ bbox, phải biến đổi tọa độ cùng phép resize. Với
resize theo hai hệ số `scale_x` và `scale_y`:

```text
x_new      = x * scale_x
y_new      = y * scale_y
width_new  = width * scale_x
height_new = height * scale_y
```

Nếu dùng letterbox, ngoài phép scale còn phải cộng phần padding vào `x_new` và
`y_new`. Nếu crop bbox trước, tọa độ mới của một điểm là tọa độ cũ trừ đi
`(left, top)`, rồi mới nhân hệ số resize.

### Dùng parts/keypoints như thế nào?

`parts/parts.txt` ánh xạ `part_id` sang tên bộ phận. `parts/part_locs.txt` ghi:

```text
image_id part_id x y visible
```

`x, y` là tọa độ điểm trên ảnh gốc, không phải tọa độ chuẩn hóa. Chỉ sử dụng
điểm khi `visible = 1`; dòng `visible = 0` có thể có tọa độ `0 0` và không được
coi đó là vị trí thật.

Nếu resize ảnh, biến đổi keypoint bằng đúng phép biến đổi của ảnh:

```text
x_new = x * scale_x
y_new = y * scale_y
```

Nếu letterbox thì cộng padding vào tọa độ. Nếu crop ảnh tại `(left, top)`, dùng
`x_crop = x - left`, `y_crop = y - top`, rồi mới resize. Điểm nằm ngoài crop
hoặc có `visible = 0` phải được đánh dấu không hợp lệ, không đưa vào loss.

Parts chỉ cần thiết khi làm part-based model, keypoint localization, attention
theo bộ phận hoặc phân tích trực quan. Chúng không phải nhãn bắt buộc của bài
toán phân loại loài.

### Quy trình nên làm trước mắt

1. Dùng `images.txt`, `image_class_labels.txt` và `train_test_split.txt` để tạo
   Dataset 555 lớp.
2. Resize/crop ảnh về `224x224`, bắt đầu bằng ảnh đầy đủ và giữ nguyên split có
   sẵn.
3. Huấn luyện baseline CNN hoặc ResNet, chưa dùng bbox và parts.
4. Chạy thêm một thí nghiệm crop bbox, giữ nguyên model, split và augmentation
   để so sánh công bằng.
5. Chỉ dùng parts nếu mục tiêu chuyển sang mô hình khai thác bộ phận; khi đó
   phải áp dụng cùng phép biến đổi hình học cho ảnh và keypoint.

## Kết quả EDA chính (số liệu thực, từ `EDA_NABirds.ipynb`)

**Tổng quan**
- 48,562 ảnh · 555 lớp lá · 1011 node phân loại (456 node cha) · 22 bộ (order) cấp cao nhất
- Split: 23,929 train / 24,633 test (≈49.3% / 50.7%) — **mỗi lớp đều có ≥1 ảnh ở cả train và test**
  (min 4 ảnh train / 9 ảnh test cho lớp ít nhất)
- 1,031 nhiếp ảnh gia khác nhau
- **Toàn vẹn dữ liệu tốt**: 0 ảnh thiếu file, 0 tham chiếu class_id sai, 0 NaN sau khi merge,
  0/300 ảnh mẫu bị lỗi khi mở bằng PIL

**Phân bố lớp (imbalance)**
- Trung bình 87.5 ảnh/lớp, min 13, max 120 → **tỉ lệ mất cân bằng 9.23x**
- 1 lớp có <20 ảnh, 7 lớp có <30 ảnh → cần class weighting / oversampling khi train 555 lớp

**Phân cấp (taxonomy)**
- "Perching Birds" (Passeriformes) áp đảo: 24,787 ảnh / 265 loài (≈51% tổng dataset)
- Các bộ nhỏ nhất: Parrots (46 ảnh, 1 loài), Nightjars (86), Storks (104)
- Depth chỉ có 2 giá trị (3 hoặc 4) — cây phân loại tương đối nông

**Ảnh & bounding box**
- Ảnh có độ phân giải cao: median 1024×683, chỉ 0.09% ảnh có cạnh <224px → resize tự do cho CNN/ResNet
- Aspect ratio trung bình 1.30 (đa số ảnh ngang)
- Bbox chiếm trung bình 30.7% diện tích ảnh (median 28.3%) — không cần crop bắt buộc trước khi train,
  nhưng có thể crop theo bbox để tăng tín hiệu nếu cần
- 201 ảnh (0.41%) có bbox vượt ra ngoài biên ảnh — lỗi annotation nhỏ, không đáng kể

**Parts (keypoints)** — 11 bộ phận, annotate đầy đủ cho mọi ảnh, không bắt buộc dùng ở 3 dự án hiện tại

**Rủi ro leakage nhiếp ảnh gia — đã kiểm tra lại, KHÔNG đáng lo**
- 610/1031 nhiếp ảnh gia (98.5% số ảnh) có ảnh ở cả train và test. Nhưng đây là con số gây hiểu lầm.
- Kiểm tra chặt hơn ở mức **(lớp, nhiếp ảnh gia)**: chỉ **49 / 22,494 cặp** bị chia đôi giữa train và
  test, tương ứng **172 ảnh (0.4%)**. Nghĩa là split gốc đã được làm *photographer-disjoint trong từng
  lớp*: một người có thể xuất hiện ở cả hai bên, nhưng gần như không bao giờ cho **cùng một loài**.
- → Model không thể học tắt "phong cách chụp của người X = loài Y". **Dùng thẳng `train_test_split.txt`,
  không cần tự xây split theo photographer.**

## Định hướng 3 dự án (dựa trên EDA)

1. **Binary (CNN/ResNet)**: dataset không có nhãn nhị phân sẵn — tự định nghĩa, ví dụ
   "Perching Birds" (51.0%) vs "Others" (49.0%, cân bằng tốt), hoặc cặp loài dễ nhầm cùng chi.
2. **Multi-class softmax (555 lớp)**: dùng đúng `train_test_split.txt`; xử lý imbalance 9.23x bằng
   class weighting/oversampling; có thể thử nhanh trên vài lớp lớn nhất trước khi scale full 555 lớp.
3. **Zero-shot**: chỉ có tên lớp dạng text + cây phân loại (không có attribute vector như CUB-200) →
   phù hợp hướng CLIP-style (text embedding), không phù hợp attribute-based ZSL cổ điển; cần tự tạo
   seen/unseen split (ví dụ theo order hiếm — thử nghiệm với Parrots/Nightjars/Storks = 3 order/236 ảnh unseen).

## Môi trường

- Thư viện EDA: `pandas`, `numpy`, `matplotlib`, `pillow` (đã cài, đủ dùng, không cần GPU)
- Huấn luyện model (bước sau): dùng conda env `deeplearning` (CUDA) với `torch`/`torchvision`

---
