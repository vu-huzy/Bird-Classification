"""C3 — chuyển giao NABirds → CUB-200-2011.

Câu hỏi: fine-tune trên NABirds (48k ảnh, 555 lớp) rồi mới sang CUB (12k ảnh,
200 lớp) có hơn đi thẳng từ ImageNet sang CUB không? Đây là chiều "nguồn lớn →
đích nhỏ" mà Cui et al. CVPR'18 dựng cả bài báo quanh nó, nhưng họ đo
ImageNet→iNaturalist→CUB, còn ở đây nguồn trung gian là chính NABirds.

Hai nhánh khác nhau ĐÚNG MỘT biến — trọng số khởi tạo của backbone:

    --init imagenet    backbone = checkpoint ImageNet gốc (đường đi thẳng)
    --init nabirds     backbone = checkpoint đã fine-tune 555 lớp NABirds

Cả hai đều thay head thành 200 lớp và train cùng recipe, cùng epoch, cùng lr.

Vì sao trước đây không chạy được: repo chỉ có 2 bản CUB *captioned* trên
HuggingFace, mỗi bản 5,994 ảnh **train** và KHÔNG có split test. Bản đầy đủ ở
`cub/CUB_200_2011/` (11,788 ảnh, 5,994/5,794) mới cho phép đánh giá.

Lưu ý về chồng lấn: 142/200 loài CUB có mặt trong NABirds (docs/02 mục 0). Nên
đây KHÔNG phải transfer sang miền mới — nó đo "thấy thêm ảnh chim có giúp không",
và phần lớn lợi ích có thể đến từ 142 loài trùng. Con số phải đọc kèm điều đó.

    python src/cub_transfer.py --model vit_b_16_in21k --init imagenet
    python src/cub_transfer.py --model vit_b_16_in21k --init nabirds \\
           --ckpt runs/vit_b_16_in21k_224/best.pt
    python src/cub_transfer.py --report
"""
import argparse
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

import models_zoo
from nabirds_data import build_transforms

REPO = models_zoo.REPO
CUB_ROOT = os.path.join(REPO, 'cub', 'CUB_200_2011')
RESULTS = os.path.join(REPO, 'results', 'cub')
NUM_CUB = 200


class CUB(Dataset):
    """CUB-200-2011 bản gốc. `train_test_split.txt`: 1 = train, 0 = test."""

    def __init__(self, split, transform, root=CUB_ROOT):
        ids, paths, labels, is_train = [], {}, {}, {}
        with open(os.path.join(root, 'images.txt'), encoding='utf-8') as f:
            for line in f:
                i, p = line.split()
                paths[i] = p
        with open(os.path.join(root, 'image_class_labels.txt'), encoding='utf-8') as f:
            for line in f:
                i, c = line.split()
                labels[i] = int(c) - 1          # file đánh số từ 1
        with open(os.path.join(root, 'train_test_split.txt'), encoding='utf-8') as f:
            for line in f:
                i, t = line.split()
                is_train[i] = t == '1'
        for i in paths:
            if is_train[i] == (split == 'train'):
                ids.append(i)
        self.samples = [(os.path.join(root, 'images', paths[i]), labels[i]) for i in ids]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, k):
        path, label = self.samples[k]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


def load_backbone(model, name, ckpt_path):
    """Nạp trọng số NABirds vào backbone, BỎ head 555 lớp.

    Head phải bỏ vì nó có shape (555, D) còn ta cần (200, D). Hàm trả về số
    tensor thực sự nạp được để đối chiếu — nếu con số này nhỏ bất thường thì
    checkpoint không khớp kiến trúc và phải dừng, chứ không chạy tiếp lặng lẽ.
    """
    sd = torch.load(ckpt_path, map_location='cpu')['model']
    head_keys = [h.rstrip('.') for h in models_zoo.head_parameter_names(model, name)]
    keep = {k: v for k, v in sd.items()
            if not any(k.startswith(h + '.') for h in head_keys)}
    missing, unexpected = model.load_state_dict(keep, strict=False)
    n_loaded = len(keep) - len(unexpected)
    total = len(model.state_dict())
    if n_loaded < 0.5 * total:
        raise ValueError(f'chi nap duoc {n_loaded}/{total} tensor tu {ckpt_path} '
                         f'-> checkpoint khong khop kien truc, dung lai')
    return n_loaded, total, len(unexpected)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(x)
        o = o.logits if hasattr(o, 'logits') else o
        correct += (o.float().argmax(1) == y).sum().item()
        n += y.numel()
    return correct / n


def report():
    import pandas as pd
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS, '*', 'summary.json'))):
        rows.append(json.load(open(f, encoding='utf-8')))
    if not rows:
        print('chua co run CUB nao')
        return
    df = pd.DataFrame(rows).sort_values('test_top1', ascending=False)
    cols = ['run', 'model', 'init', 'img_size', 'epochs_run', 'test_top1',
            'best_val_top1', 'train_minutes']
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    df.to_csv(os.path.join(RESULTS, 'comparison.csv'), index=False, encoding='utf-8-sig')
    print(f'\n-> results/cub/comparison.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='vit_b_16_in21k', choices=models_zoo.MODEL_NAMES)
    ap.add_argument('--init', default='imagenet', choices=['imagenet', 'nabirds'])
    ap.add_argument('--ckpt', default=None, help='bắt buộc khi --init nabirds')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--img-size', type=int, default=None)
    ap.add_argument('--lr-backbone', type=float, default=None)
    ap.add_argument('--lr-head', type=float, default=None)
    ap.add_argument('--optimizer', default='sgd', choices=['sgd', 'adamw'])
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.report:
        return report()
    if a.init == 'nabirds' and not a.ckpt:
        ap.error('--init nabirds phai kem --ckpt')

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = 'cuda'

    model, default_size = models_zoo.build_model(a.model, img_size=a.img_size)
    img_size = a.img_size or default_size
    mean, std = model.norm_mean, model.norm_std
    if a.init == 'nabirds':
        models_zoo.set_num_classes(model, a.model, models_zoo.NUM_CLASSES)
        n_loaded, total, unexpected = load_backbone(model, a.model, a.ckpt)
        print(f'  nap {n_loaded}/{total} tensor tu {a.ckpt} '
              f'({unexpected} key thua)', flush=True)
    models_zoo.set_num_classes(model, a.model, NUM_CUB)
    model = model.to(device).to(memory_format=torch.channels_last)

    run = a.run_name or f'{a.model}_{img_size}_{a.init}'
    tr = CUB('train', build_transforms(img_size, True, 'standard', mean, std))
    te = CUB('test', build_transforms(img_size, False, mean=mean, std=std))
    mk = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=a.batch_size, shuffle=sh, drop_last=sh, num_workers=a.workers,
        pin_memory=True, persistent_workers=a.workers > 0)
    tr_loader, te_loader = mk(tr, True), mk(te, False)
    print(f'== cub/{run} == train {len(tr)} | test {len(te)} | img {img_size} '
          f'| init {a.init}', flush=True)

    from train import DEFAULT_LR
    lr_b, lr_h = DEFAULT_LR[a.model]
    lr_b = a.lr_backbone if a.lr_backbone is not None else lr_b
    lr_h = a.lr_head if a.lr_head is not None else lr_h
    groups = models_zoo.param_groups(model, a.model, lr_b, lr_h, a.weight_decay)
    opt = (torch.optim.SGD(groups, momentum=0.9, nesterov=True) if a.optimizer == 'sgd'
           else torch.optim.AdamW(groups))
    steps = a.epochs * len(tr_loader)
    warm = min(200, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / max(1, warm) if s < warm else
        0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm))))))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    # CUB không có split val riêng và tập test chỉ 5,794 ảnh. Tách val từ train
    # sẽ làm train nhỏ đi 15% trên một dataset vốn đã nhỏ. Ở đây chạy đủ số epoch
    # cố định rồi báo cáo epoch CUỐI — không chọn checkpoint theo test.
    # Đó là điều kiện để so hai nhánh cho công bằng, và phải ghi rõ ra.
    best_seen, t0 = 0.0, time.time()
    for ep in range(a.epochs):
        model.train()
        tot = corr = seen = 0
        for x, y in tr_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = model(x)
                o = o.logits if hasattr(o, 'logits') else o
                loss = crit(o, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * y.numel()
            corr += (o.argmax(1) == y).sum().item()
            seen += y.numel()
        acc = evaluate(model, te_loader, device)
        best_seen = max(best_seen, acc)
        print(f'  ep {ep+1:02d}/{a.epochs}  loss {tot/seen:.3f}  '
              f'train-acc {corr/seen*100:5.2f}  |  test {acc*100:5.2f}', flush=True)
    minutes = (time.time() - t0) / 60

    os.makedirs(os.path.join(RESULTS, run), exist_ok=True)
    with open(os.path.join(RESULTS, run, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'run': run, 'model': a.model, 'init': a.init, 'ckpt': a.ckpt,
                   'img_size': img_size, 'epochs_run': a.epochs,
                   'test_top1': round(acc * 100, 2),
                   'best_test_seen': round(best_seen * 100, 2),
                   'n_train': len(tr), 'n_test': len(te),
                   'train_minutes': round(minutes, 1)}, f, indent=2)
    print(f'  -> results/cub/{run}/summary.json  (epoch cuoi {acc*100:.2f}, '
          f'cao nhat tung thay {best_seen*100:.2f})', flush=True)


if __name__ == '__main__':
    main()
