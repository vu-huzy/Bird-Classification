"""Bước 4 — gom mọi `summary.json` trong `zeroshot/results/` thành bảng báo cáo.

    python zeroshot/src/report_zs.py            # in ra màn hình
    python zeroshot/src/report_zs.py --save     # ghi thêm CSV + Markdown
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zs_env  # noqa: E402

PCT = 100
SEP = '=' * 118


def load_all():
    rows = []
    for p in sorted(glob.glob(os.path.join(zs_env.RESULTS_DIR, '*', 'summary.json'))):
        with open(p, encoding='utf-8') as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)


def table_zeroshot(df):
    """Bước 2 — zero-shot thuần, không train gì."""
    d = df[df.run.str.contains('_T') & ~df.run.str.startswith(('mets_', 'lora_', 'cub_'))]
    leaf = d[d.space == 'leaf'].copy()
    sp = d[d.space == 'species'].set_index(['model', 'level']).top1_accuracy
    leaf['sp404_direct'] = [sp.get((m, l), float('nan'))
                            for m, l in zip(leaf.model, leaf.level)]
    cols = ['model', 'level', 'top1_accuracy', 'top5_accuracy', 'sp404_direct',
            'species_404_accuracy', 'macro_f1', 'vp_bal', 'vp81_bal', 'vp_chance']
    t = leaf[cols].copy()
    for c in cols[2:]:
        t[c] = (PCT * t[c]).round(2)
    return t.rename(columns={
        'top1_accuracy': 'leaf555_top1', 'top5_accuracy': 'leaf555_top5',
        'sp404_direct': 'sp404_top1', 'species_404_accuracy': 'leaf->sp404',
        'macro_f1': 'macroF1', 'vp_bal': 'var137bal', 'vp81_bal': 'var81bal',
        'vp_chance': 'chance'}).sort_values(['model', 'var81bal'],
                                            ascending=[True, False])


def table_mets(df):
    """Bước 3a (B1) — METS-analogue trên embedding đóng băng. `b0_*` = không train."""
    d = df[df.run.str.startswith('mets_')].copy()
    if d.empty:
        return d
    keep = ['split', 'img', 'txt', 'level', 'seed', 'zsl_unseen_acc', 'gzsl_seen',
            'gzsl_unseen', 'gzsl_harmonic', 'all555_top1', 'vp81_bal', 'vp_bal',
            'b0_zsl_unseen_acc', 'b0_gzsl_harmonic', 'b0_vp81_bal', 'b0_all555_top1']
    t = d[[c for c in keep if c in d.columns]].copy()
    for c in t.columns:
        if t[c].dtype.kind == 'f':
            t[c] = (PCT * t[c]).round(2)
    return t.sort_values(['split', 'img', 'txt', 'level'])


def table_lora(df):
    """Bước 3b (B2) — LoRA trên image tower, so với B1 cùng cấu hình."""
    d = df[df.run.str.startswith('lora_')].copy()
    if d.empty:
        return d
    keep = ['split', 'img', 'txt', 'level', 'lora_params_M', 'epochs', 'peak_vram_GB',
            'minutes', 'zsl_unseen_acc', 'gzsl_harmonic', 'vp81_bal', 'all555_top1']
    t = d[[c for c in keep if c in d.columns]].copy()
    for c in ('zsl_unseen_acc', 'gzsl_harmonic', 'vp81_bal', 'all555_top1'):
        if c in t:
            t[c] = (PCT * t[c]).round(2)
    return t.round(2).sort_values(['split', 'img', 'txt'])


def table_cub(df):
    """Bước 3c (B4) — contrastive trên CUB. Biến độc lập: độ hạt của caption."""
    d = df[df.run.str.startswith('cub_')].copy()
    if d.empty:
        return d
    keep = ['caps', 'lr', 'uniq_texts', 'cub_val_unseen_acc', 'zsl352_top1',
            'b0_zsl352_top1', 'vpun_bal', 'b0_vpun_bal', 'all555_top1',
            'b0_all555_top1']
    t = d[[c for c in keep if c in d.columns]].copy()
    for c in t.columns:                                  # chỉ đổi cột tỉ lệ sang %
        if any(k in c for k in ('acc', 'top1', 'bal')):
            t[c] = (PCT * t[c]).round(2)
    t['uniq_texts'] = t.uniq_texts.astype(int)
    return t.sort_values(['caps', 'lr'])


TABLES = [
    (table_zeroshot, 'zeroshot',
     'BUOC 2 - ZERO-SHOT THUAN (khong train gi). var*bal = cho san dung LOAI, '
     'chon dung BIEN THE, can bang theo la.'),
    (table_mets, 'mets',
     'BUOC 3a (B1) - METS-ANALOGUE tren embedding dong bang. b0_* = KHONG train.'),
    (table_lora, 'lora',
     'BUOC 3b (B2) - LoRA tren image tower.'),
    (table_cub, 'cub',
     'BUOC 3c (B4) - contrastive tren CUB. caps naive = text muc LOP (200 chuoi), '
     'cupl = text BIEN THIEN (4490 chuoi).'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', action='store_true')
    args = ap.parse_args()
    df = load_all()

    md = []
    for fn, name, title in TABLES:
        t = fn(df)
        if t.empty:
            continue
        print(SEP)
        print(title)
        print(SEP)
        print(t.to_string(index=False))
        print()
        md.append((name, title, t))

    if args.save:
        for name, title, t in md:
            t.to_csv(os.path.join(zs_env.RESULTS_DIR, f'table_{name}.csv'), index=False)
        with open(os.path.join(zs_env.RESULTS_DIR, 'tables.md'), 'w',
                  encoding='utf-8') as f:
            for name, title, t in md:
                f.write(f'## {title}\n\n{t.to_markdown(index=False)}\n\n')
        print(f'-> {zs_env.RESULTS_DIR}: ' +
              ', '.join(f'table_{n}.csv' for n, _, _ in md) + ', tables.md')


if __name__ == '__main__':
    main()
