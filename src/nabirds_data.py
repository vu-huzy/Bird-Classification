"""NABirds Dataset cho bài toán 555 lớp (giao thức chuẩn của các paper).

Giao thức:
  - dùng nguyên `train_test_split.txt` gốc (23,929 train / 24,633 test)
  - KHÔNG dùng bounding box lúc test (giống TransFG / MetaFormer / MPSA)
  - `--use-bbox` chỉ để chạy thí nghiệm đối chứng, không phải setting chính
"""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import nabirds_io as nio

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class NABirds(Dataset):
    """555-way fine-grained classification.

    Trả về (image_tensor, label) với label trong [0, 554].
    """

    def __init__(self, root=nio.DEFAULT_ROOT, split='train', transform=None,
                 image_dir='images_r448', use_bbox=False, bbox_margin=0.15,
                 indices=None, paste_p=0.0, paste_pool=None, paste_scale=(0.3, 0.7),
                 paste_mean=IMAGENET_MEAN, paste_std=IMAGENET_STD):
        self.root = root
        self.transform = transform
        self.use_bbox = use_bbox
        self.bbox_margin = bbox_margin

        img_root = os.path.join(root, image_dir)
        if not os.path.isdir(img_root):
            img_root = os.path.join(root, 'images')
        self.img_root = img_root
        self.orig_root = os.path.join(root, 'images')

        rel_paths = nio.load_image_paths(root)
        labels = nio.load_image_labels(root)
        train_ids, test_ids = nio.load_train_test_split(root)
        if split not in ('train', 'test'):
            raise ValueError(f'split must be train/test, got {split}')
        ids = train_ids if split == 'train' else test_ids

        self.label_index, self.leaf_ids = nio.build_label_index(labels)
        self.class_names = nio.load_class_names(root)
        self.hierarchy = nio.load_hierarchy(root)
        self.taxonomy = nio.build_taxonomy(self.leaf_ids, self.hierarchy, self.class_names)

        self.sizes = nio.load_image_sizes(root) if use_bbox else None
        self.bboxes = nio.load_bounding_box_annotations(root) if use_bbox else None

        self.samples = [(i, rel_paths[i], self.label_index[labels[i]]) for i in ids]
        if indices is not None:
            self.samples = [self.samples[k] for k in indices]
        self.split = split

        # --- CMO: dán chim của LỚP HIẾM lên ảnh của lớp nhiều ảnh -------------
        # Park et al., CVPR 2022 — "The Majority Can Help The Minority". Ý tưởng:
        # lớp thiểu số thiếu BỐI CẢNH đa dạng, nên mượn nền của lớp đa số.
        # Khác bản gốc ở một điểm quan trọng: CMO cắt một Ô NGẪU NHIÊN từ ảnh
        # thiểu số (nên có thể trượt mất con chim), còn ở đây repo có BBOX THẬT
        # cho mọi ảnh nên dán được đúng con chim.
        self.paste_p = paste_p
        self.paste_pool = paste_pool or []
        self.paste_scale = paste_scale
        if paste_p > 0:
            if not self.paste_pool:
                raise ValueError('paste_p > 0 nhung paste_pool rong')
            # bbox cần cho việc cắt, dù use_bbox=False ở nhánh chính
            self.sizes = self.sizes or nio.load_image_sizes(root)
            self.bboxes = self.bboxes or nio.load_bounding_box_annotations(root)
            # Miếng dán đã được resize thủ công nên chỉ cần chuẩn hoá, KHÔNG
            # thêm phép hình học nào nữa.
            self.transform_paste = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize(paste_mean, paste_std)])

    # -- tên lớp theo chỉ số 0..554, dùng cho bảng kết quả ------------------
    @property
    def idx_to_name(self):
        return [self.class_names[c] for c in self.leaf_ids]

    @property
    def idx_to_species(self):
        return [self.taxonomy[c][0] for c in self.leaf_ids]

    @property
    def idx_to_order(self):
        return [self.taxonomy[c][1] for c in self.leaf_ids]

    def __len__(self):
        return len(self.samples)

    def _crop_bbox(self, img, image_id):
        """Crop theo bbox đã nới biên. bbox nằm trong toạ độ ảnh GỐC nên phải
        scale theo tỉ lệ ảnh đã pre-resize."""
        ow, _ = self.sizes[image_id]
        x, y, w, h = self.bboxes[image_id]
        s = img.width / ow                       # ảnh giữ nguyên aspect ratio
        x, y, w, h = x * s, y * s, w * s, h * s
        mx, my = w * self.bbox_margin, h * self.bbox_margin
        left = max(0, int(x - mx))
        top = max(0, int(y - my))
        right = min(img.width, int(x + w + mx))
        bottom = min(img.height, int(y + h + my))
        if right - left < 8 or bottom - top < 8:  # bbox hỏng -> giữ ảnh đầy đủ
            return img
        return img.crop((left, top, right, bottom))

    def _load(self, k):
        image_id, rel, label = self.samples[k]
        path = os.path.join(self.img_root, rel)
        if not os.path.exists(path):
            path = os.path.join(self.orig_root, rel)
        return Image.open(path).convert('RGB'), image_id, label

    def __getitem__(self, i):
        img, image_id, label = self._load(i)
        if self.use_bbox:
            img = self._crop_bbox(img, image_id)
        if self.transform is not None:
            img = self.transform(img)
        # `getattr` chứ không phải `self.paste_p`: trên Windows, DataLoader worker
        # được SPAWN — nó import lại module (code MỚI) rồi unpickle object dataset
        # (đã dựng bằng code CŨ). Nếu sửa file nguồn trong lúc một sweep đang chạy
        # thì object cũ thiếu thuộc tính mới và worker chết giữa chừng. Đã mất một
        # run vì đúng lỗi này (`convnext_tiny_in22k_224_noerase`, exit 1).
        if getattr(self, 'paste_p', 0.0) <= 0:
            return img, label

        # Dán SAU khi transform (trên tensor). Nếu dán trước thì
        # RandomResizedCrop có thể cắt mất đúng con chim vừa dán -> nhãn thành
        # nhiễu, đúng điểm yếu mà SnapMix chỉ ra ở CutMix.
        import random
        if random.random() >= self.paste_p:
            return img, label, label, 1.0
        src, src_id, src_label = self._load(random.choice(self.paste_pool))
        crop = self._crop_bbox(src, src_id)
        H, W = img.shape[1], img.shape[2]
        r = random.uniform(*self.paste_scale)
        h2, w2 = max(8, int(H * r)), max(8, int(W * r))
        patch = self.transform_paste(crop.resize((w2, h2), Image.BILINEAR))
        y0, x0 = random.randint(0, H - h2), random.randint(0, W - w2)
        img = img.clone()
        img[:, y0:y0 + h2, x0:x0 + w2] = patch
        lam = 1.0 - (h2 * w2) / (H * W)      # trọng số còn lại của nhãn GỐC
        return img, label, src_label, lam


def build_transforms(img_size, train, aug='standard', mean=IMAGENET_MEAN,
                     std=IMAGENET_STD, erasing_p=0.25, hue=0.02, rotate=0.0):
    """Augmentation cho FGVC.

    Khác với recipe ImageNet mặc định ở 2 điểm quan trọng:
      - RandomResizedCrop scale=(0.3, 1.0) thay vì (0.08, 1.0). Mặc định quá
        mạnh, cắt mất con chim (bbox chỉ chiếm median 28.3% diện tích ảnh).
      - hue jitter ~ 0. Màu bộ lông CHÍNH LÀ nhãn (Yellow vs Orange Warbler),
        xoay hue là phá nhãn.
    """
    if train:
        ops = [
            transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0),
                                         ratio=(3 / 4, 4 / 3)),
            transforms.RandomHorizontalFlip(0.5),
        ]
        # `rotate` mặc định 0: chim có hướng chuẩn, xoay mạnh là phá nhãn. Nếu
        # bật thì giữ <= 15 độ (PLAN mục E).
        if rotate > 0:
            ops.append(transforms.RandomRotation(rotate))
        if aug == 'standard':
            ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                              saturation=0.1, hue=hue))
        ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        # erasing_p=0 -> tắt hẳn. Đây là ablation E1: tài liệu ghi Cutout làm
        # ResNet-50 mất ~2 điểm trên CUB vì có xác suất xoá TRÚNG vùng phân biệt,
        # mà vùng phân biệt của chim rất nhỏ. Repo bật 0.25 từ đầu, chưa ai kiểm.
        if aug == 'standard' and erasing_p > 0:
            ops.append(transforms.RandomErasing(p=erasing_p, scale=(0.02, 0.15)))
        return transforms.Compose(ops)

    resize = int(round(img_size * 1.14))         # 224 -> 256, 299 -> 341
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def stratified_val_split(samples, val_frac, seed=0, min_class_size=20):
    """Tách val ngẫu nhiên khỏi train, phân tầng theo lớp.

    Quy tắc quan trọng: **lớp ít ảnh được ưu tiên cho train**. Lớp có dưới
    `min_class_size` ảnh train sẽ KHÔNG đóng góp ảnh nào cho val — giữ trọn
    100% dữ liệu để học. Phân bố ảnh train/lớp ở NABirds là min 4, p5 17,
    median 44, nên ngưỡng 20 loại đúng nhóm đuôi dài (49/555 lớp) mà chỉ làm
    val nhỏ đi vài phần trăm.

    Val chỉ dùng để chọn checkpoint và early stopping; con số báo cáo cuối cùng
    luôn tính trên tập test gốc nên không bị ảnh hưởng bởi lựa chọn này.

    Lưu ý khi đọc log: val-top1 thường CAO hơn test-top1 vài điểm vì val lấy
    ngẫu nhiên từ cùng phân phối train, còn test là split riêng của dataset.
    Điều đó bình thường — cái cần theo dõi là val có TĂNG đều hay không, không
    phải giá trị tuyệt đối của nó.
    """
    import collections
    import random
    by_cls = collections.defaultdict(list)
    for k, (_, _, lab) in enumerate(samples):
        by_cls[lab].append(k)
    rng = random.Random(seed)
    val_idx = []
    for lab, ks in by_cls.items():
        if len(ks) < min_class_size:      # lớp hiếm -> dồn hết cho train
            continue
        rng.shuffle(ks)
        n_val = min(int(round(len(ks) * val_frac)), max(0, len(ks) - 2))
        val_idx.extend(ks[:n_val])
    val_set = set(val_idx)
    train_idx = [k for k in range(len(samples)) if k not in val_set]
    return sorted(train_idx), sorted(val_idx)


def build_loaders(img_size, batch_size, workers=8, use_bbox=False, aug='standard',
                  root=nio.DEFAULT_ROOT, val_frac=0.0, seed=0,
                  image_dir='images_r448', val_min_class=20,
                  mean=IMAGENET_MEAN, std=IMAGENET_STD,
                  erasing_p=0.25, hue=0.02, rotate=0.0,
                  cmo_p=0.0, cmo_tail_max=30):
    """Trả về (train_set, val_set, test_set, train_loader, val_loader, test_loader).

    `val_frac > 0` tách một phần tập train ra làm validation. Val dùng để chọn
    checkpoint tốt nhất và early stopping; test CHỈ được chạy một lần ở cuối.
    Nếu chọn checkpoint bằng chính test thì con số báo cáo sẽ lạc quan có hệ
    thống — đó là lỗi phương pháp dễ bị phản biện nhất khi so với paper.
    """
    mk = lambda sp, tf, idx=None: NABirds(root, sp, tf, image_dir=image_dir,
                                         use_bbox=use_bbox, indices=idx)
    # `mean`/`std` phải theo MODEL: BioCLIP pretrain bằng chuẩn hoá CLIP, đưa ảnh
    # chuẩn hoá kiểu ImageNet vào là lệch phân phối đầu vào ngay từ epoch 0.
    tf_train = build_transforms(img_size, True, aug, mean, std,
                                erasing_p=erasing_p, hue=hue, rotate=rotate)
    tf_eval = build_transforms(img_size, False, mean=mean, std=std)
    full_train = mk('train', tf_train)
    if val_frac > 0:
        tr_idx, va_idx = stratified_val_split(full_train.samples, val_frac, seed,
                                              min_class_size=val_min_class)
        train_set = mk('train', tf_train, tr_idx)
        val_set = mk('train', tf_eval, va_idx)
    else:
        tr_idx, train_set, val_set = list(range(len(full_train.samples))), full_train, None

    if cmo_p > 0:
        # Pool nguồn = mọi ảnh thuộc lớp có < cmo_tail_max ảnh TRAIN. Chỉ số phải
        # tính TRÊN train_set (đã trừ val), không phải trên full_train.
        import collections
        cnt = collections.Counter(lab for _, _, lab in train_set.samples)
        pool = [k for k, (_, _, lab) in enumerate(train_set.samples)
                if cnt[lab] < cmo_tail_max]
        train_set = NABirds(root, 'train', tf_train, image_dir=image_dir,
                            use_bbox=use_bbox, indices=tr_idx, paste_p=cmo_p,
                            paste_pool=pool, paste_mean=mean, paste_std=std)
        n_cls = len({train_set.samples[k][2] for k in pool})
        print(f'  CMO: pool {len(pool)} anh tu {n_cls} lop co < {cmo_tail_max} '
              f'anh train, xac suat dan {cmo_p}', flush=True)
    test_set = mk('test', tf_eval)
    # Mỗi worker chiếm ~765 MB RSS (import torch). Nếu để train và test cùng
    # `persistent_workers=True` với 8 worker thì có 16 process sống song song
    # (~12 GB) -> trên máy 32 GB sẽ hết RAM, hệ thống paging và GPU bị bỏ đói.
    # Vì vậy: train giữ worker sống, test dùng ít worker và tắt persistent.
    # Val/test khong phai nut that (chi 3,510 / 24,633 anh, forward-only) nen
    # gioi han cung o 4 worker: de danh RAM cho train loader.
    test_workers = max(2, min(4, workers // 2))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None)
    def eval_loader(ds, persistent):
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=test_workers, pin_memory=True,
            persistent_workers=persistent and test_workers > 0,
            prefetch_factor=4 if test_workers > 0 else None)

    # Val chạy MỖI epoch -> phải giữ worker sống. Đo trên tập val 3,510 ảnh:
    #   persistent=0, workers=3 -> 31.3s / lượt   (spawn lại + import torch lại)
    #   persistent=1, workers=3 ->  7.2s / lượt
    #   persistent=1, workers=6 ->  3.4s / lượt
    # Chênh ~13x, tương ứng ~30s lãng phí mỗi epoch nếu để non-persistent.
    # Test chỉ chạy MỘT lần ở cuối -> không cần giữ worker, tiết kiệm RAM.
    test_loader = eval_loader(test_set, persistent=False)
    val_loader = eval_loader(val_set, persistent=True) if val_set is not None else None
    return train_set, val_set, test_set, train_loader, val_loader, test_loader
