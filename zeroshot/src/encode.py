"""Bước 2a — mã hoá ảnh NABirds một lần cho mỗi model, cache ra `.npy`.

Đây là thứ làm cho toàn bộ bước 2 gần như miễn phí: sau khi có embedding ảnh
(24,633 x D cho test), thử một biến thể prompt chỉ còn là một phép nhân ma trận
`(24633, D) @ (D, 555)` — tính bằng giây. Không phải chạy lại model.

Chạy:
    python zeroshot/src/encode.py --model bioclip --split test
    python zeroshot/src/encode.py --model bioclip --split train
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

import zs_env  # noqa: E402
import zs_models  # noqa: E402
from nabirds_data import NABirds  # noqa: E402

EMB_DIR = os.path.join(zs_env.CACHE_DIR, 'emb')


def paths(model, split):
    os.makedirs(EMB_DIR, exist_ok=True)
    return (os.path.join(EMB_DIR, f'{model}_{split}.npy'),
            os.path.join(EMB_DIR, f'labels_{split}.npy'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(zs_models.REGISTRY))
    ap.add_argument('--split', default='test', choices=['train', 'test'])
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    emb_path, lab_path = paths(args.model, args.split)
    if os.path.exists(emb_path):
        print(f'da co {emb_path} -> bo qua')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess, _ = zs_models.load(args.model, device)
    print(f'{args.model}: {preprocess}')

    ds = NABirds(split=args.split, transform=preprocess, image_dir='images_r448')
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=False)
    print(f'{args.split}: {len(ds):,} anh')

    feats, labels, t0 = [], [], time.time()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i, (img, y) in enumerate(loader):
            f = model.encode_image(img.to(device, non_blocking=True))
            feats.append(torch.nn.functional.normalize(f, dim=-1).float().cpu())
            labels.append(y)
            if i % 20 == 0:
                done = (i + 1) * args.batch_size
                print(f'  {min(done, len(ds)):6d}/{len(ds)}  '
                      f'{done/(time.time()-t0):7.1f} img/s', flush=True)

    feats = torch.cat(feats).numpy().astype(np.float32)
    labels = torch.cat(labels).numpy().astype(np.int64)
    np.save(emb_path, feats)
    np.save(lab_path, labels)
    print(f'\n{feats.shape} trong {time.time()-t0:.0f}s -> {emb_path}')
    print(f'  |f| trung binh = {np.linalg.norm(feats, axis=1).mean():.4f} (can ~1.0)')


if __name__ == '__main__':
    main()
