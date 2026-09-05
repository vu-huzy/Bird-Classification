# project-dl — NABirds Deep Learning Project

Dataset: **NABirds v0** (Cornell Lab of Ornithology / Visipedia) — ảnh chim Bắc Mỹ, phục vụ 3 dự án:

1. **Binary classification** — CNN / ResNet cơ bản
2. **Multi-class classification** — softmax, 555 lớp (single-label, không phải multi-label thật sự)
3. **Zero-shot classification**

Môi trường huấn luyện: conda `deeplearning` (CUDA) — chưa dùng ở bước EDA này.

## Trạng thái hiện tại

- [x] EDA đầy đủ dataset → [EDA_NABirds.ipynb](EDA_NABirds.ipynb) (đã chạy, có sẵn output/biểu đồ)
- [ ] Dự án #1: Binary CNN/ResNet — chưa bắt đầu
- [ ] Dự án #2: Multi-class softmax (555 lớp) — chưa bắt đầu
- [ ] Dự án #3: Zero-shot — chưa bắt đầu

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

**Rủi ro leakage nhiếp ảnh gia**
- 610/1031 nhiếp ảnh gia (98.5% số ảnh) có ảnh ở **cả train và test** → cần lưu ý khi diễn giải
  kết quả benchmark (model có thể học nhẹ "phong cách chụp" thay vì đặc điểm loài)

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
