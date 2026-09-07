"""C5 — logit adjustment hậu kiểm cho lớp đuôi. KHÔNG train lại gì.

Menon et al., *Long-tail learning via logit adjustment*, ICLR 2021: model train
bằng cross-entropy thường học p(y|x) vốn đã nghiêng theo tần suất lớp. Muốn tối
ưu sai số CÂN BẰNG thì lúc suy luận phải trừ đi log tiên nghiệm:

    argmax_y [ f_y(x) - tau * log pi_y ]

`pi_y` = tần suất lớp y trong tập TRAIN THẬT (train trừ val, đúng tập model đã
thấy). `tau = 0` là không đổi gì; `tau = 1` là quy tắc tối ưu cho balanced error.

Repo đã ĐO chênh 18 điểm F1 giữa lớp đuôi và lớp đầu nhưng chưa SỬA. Đây là bản
sửa rẻ nhất: chỉ là phép trừ trên logits đã cache bởi `src/ensemble.py`.

`tau` được chọn trên VAL rồi mới báo cáo trên TEST — chọn tau bằng test là cùng
loại gian lận với bẫy #11.

    python src/ensemble.py --cache      # phải chạy trước để có logits
    python src/logit_adjust.py
"""
import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import models_zoo
from aug_report import EXCLUDE, N_HARD, TAIL_TRAIN_MAX, hard_classes
from nabirds_data import NABirds, stratified_val_split

REPO = models_zoo.REPO
RESULTS = os.path.join(REPO, 'results')
RUNS = os.path.join(REPO, 'runs')
TAUS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5]


def train_prior():
    """Tần suất lớp trên tập train THẬT (đã trừ val), khớp đúng tập model đã thấy."""
    ds = NABirds(split='train')
    tr_idx, _ = stratified_val_split(ds.samples, 0.15, seed=0, min_class_size=20)
    cnt = collections.Counter(ds.samples[k][2] for k in tr_idx)
    pi = np.array([cnt.get(c, 0) for c in range(models_zoo.NUM_CLASSES)], dtype=np.float64)
    pi = np.maximum(pi, 1.0)                      # tránh log(0) cho lớp rỗng
    return pi / pi.sum(), ds


def f1_per_class(y, pred, n_cls):
    f1 = np.zeros(n_cls)
    for c in range(n_cls):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f1[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='plain', help='biến thể TTA dùng logits nào')
    a = ap.parse_args()

    pi, ds = train_prior()
    log_pi = np.log(pi)[None, :]
    names = np.array(ds.idx_to_name)

    yv_path = os.path.join(RUNS, 'labels_val.npy')
    yt_path = os.path.join(RUNS, 'labels_test.npy')
    if not (os.path.exists(yv_path) and os.path.exists(yt_path)):
        print('chua co logits cache. Chay truoc: python src/ensemble.py --cache')
        return
    yv, yt = np.load(yv_path), np.load(yt_path)

    tail_names = set()
    cnt = collections.Counter(lab for _, _, lab in ds.samples)
    for c in range(len(names)):
        if cnt[c] < TAIL_TRAIN_MAX:
            tail_names.add(names[c])
    pc_paths = [p for p in sorted(glob.glob(os.path.join(RESULTS, '*', 'per_class.csv')))
                if not os.path.basename(os.path.dirname(p)).startswith('_')]
    hard_set, _ = hard_classes(pc_paths)
    is_tail = np.array([n in tail_names for n in names])
    is_hard = np.array([n in hard_set for n in names])

    rows = []
    for d in sorted(os.listdir(RUNS)):
        fv = os.path.join(RUNS, d, f'logits_val_{a.tag}.npy')
        ft = os.path.join(RUNS, d, f'logits_test_{a.tag}.npy')
        if not (os.path.exists(fv) and os.path.exists(ft)):
            continue
        lv = np.load(fv).astype(np.float32)
        lt = np.load(ft).astype(np.float32)
        # chọn tau trên VAL
        va = {t: ((lv - t * log_pi).argmax(1) == yv).mean() for t in TAUS}
        tau = max(va, key=va.get)
        base_pred, adj_pred = lt.argmax(1), (lt - tau * log_pi).argmax(1)
        f1_b, f1_a = (f1_per_class(yt, p, len(names)) for p in (base_pred, adj_pred))
        rows.append({
            'run': d, 'tau*': tau,
            'top1': round((base_pred == yt).mean() * 100, 2),
            'top1_adj': round((adj_pred == yt).mean() * 100, 2),
            'mF1': round(f1_b.mean() * 100, 2),
            'mF1_adj': round(f1_a.mean() * 100, 2),
            'F1tail': round(f1_b[is_tail].mean() * 100, 2),
            'F1tail_adj': round(f1_a[is_tail].mean() * 100, 2),
            'F1hard': round(f1_b[is_hard].mean() * 100, 2),
            'F1hard_adj': round(f1_a[is_hard].mean() * 100, 2),
        })
    if not rows:
        print(f'khong tim thay logits_*_{a.tag}.npy. Chay: python src/ensemble.py --cache')
        return
    df = pd.DataFrame(rows).sort_values('top1', ascending=False)
    df['d_top1'] = (df['top1_adj'] - df['top1']).round(2)
    df['d_mF1'] = (df['mF1_adj'] - df['mF1']).round(2)
    df['d_F1tail'] = (df['F1tail_adj'] - df['F1tail']).round(2)
    df['d_F1hard'] = (df['F1hard_adj'] - df['F1hard']).round(2)
    print(f'\nLogit adjustment hau kiem, tau chon tren VAL, bao cao tren TEST')
    print(f'  {int(is_tail.sum())} lop duoi (<{TAIL_TRAIN_MAX} anh train), '
          f'{int(is_hard.sum())} lop kho\n')
    print(df[['run', 'tau*', 'top1', 'd_top1', 'mF1', 'd_mF1',
              'F1tail', 'd_F1tail', 'F1hard', 'd_F1hard']].to_string(index=False))
    out = os.path.join(RESULTS, 'logit_adjust')
    df.to_csv(out + '.csv', index=False, encoding='utf-8-sig')
    with open(out + '.md', 'w', encoding='utf-8') as f:
        f.write(df.to_markdown(index=False))
    print(f'\n-> results/logit_adjust.csv|.md')


if __name__ == '__main__':
    main()
