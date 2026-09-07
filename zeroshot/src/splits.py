"""Hai split seen/unseen cho bước 3 (METS-analogue).

`variant` — trục zero-shot SẠCH của dự án này.
    Unseen = các lá "female/immature" của những loài mà lá "male" vẫn nằm trong train.
    BioCLIP được giám sát HOÀN TOÀN bằng taxonomy nên chưa từng học nhãn giới tính /
    tuổi / bộ lông. Loài thì nó đã thấy (403/404, xem bước 1), nhưng *biến thể* thì
    chưa -> đây là nhãn mới thật sự.

`species` — giao thức ZSL quen thuộc, để đối chiếu.
    Giữ ~20% loài hoàn toàn khỏi train (phân tầng theo 22 order của NABirds để không
    mất nguyên một bộ). Lưu ý trung thực: 99.8% loài NABirds đã nằm trong pretrain
    của BioCLIP -> đây là "in-domain transfer", KHÔNG phải zero-shot thật.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zs_env  # noqa: E402


def _ordered(df):
    return df.sort_values('class_id', key=lambda s: s.astype(int)).reset_index(drop=True)


def variant_split(df):
    """-> mask (555,) True = lá unseen."""
    sub = _ordered(df)
    male_sp = set(sub.loc[sub.male_only, 'species_class_id'])
    return (sub.female_side & sub.species_class_id.isin(male_sp)).to_numpy()


def species_split(df, frac=0.2, seed=0):
    """Giữ `frac` số LOÀI khỏi train, phân tầng theo order tiếng Anh của NABirds."""
    sub = _ordered(df)
    sp = sub.drop_duplicates('species_class_id')[['species_class_id', 'order_name']]
    rng = np.random.default_rng(seed)
    held = []
    for _, g in sp.groupby('order_name'):
        ids = g.species_class_id.to_numpy()
        k = max(1, int(round(frac * len(ids)))) if len(ids) > 1 else 0
        held += list(rng.choice(ids, size=k, replace=False))
    return sub.species_class_id.isin(set(held)).to_numpy()


def describe(name, unseen_mask, df, y_train, y_test):
    sub = _ordered(df)
    n_tr = np.bincount(y_train, minlength=555)
    n_te = np.bincount(y_test, minlength=555)
    seen = ~unseen_mask
    return {
        'split': name,
        'unseen_leaves': int(unseen_mask.sum()),
        'seen_leaves': int(seen.sum()),
        'unseen_species': int(sub.loc[unseen_mask, 'species_class_id'].nunique()),
        'train_imgs_seen': int(n_tr[seen].sum()),
        'train_imgs_dropped': int(n_tr[unseen_mask].sum()),
        'test_imgs_unseen': int(n_te[unseen_mask].sum()),
        'test_imgs_seen': int(n_te[seen].sum()),
    }


def main():
    import prompts
    df = prompts.load_tables()
    y_tr = np.load(os.path.join(zs_env.CACHE_DIR, 'emb', 'labels_train.npy'))
    y_te = np.load(os.path.join(zs_env.CACHE_DIR, 'emb', 'labels_test.npy'))

    rows, masks = [], {}
    for name, mask in (('variant', variant_split(df)),
                       ('species', species_split(df))):
        masks[name] = mask
        rows.append(describe(name, mask, df, y_tr, y_te))
    out = os.path.join(zs_env.DATA_DIR, 'splits.npz')
    np.savez(out, **masks)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
