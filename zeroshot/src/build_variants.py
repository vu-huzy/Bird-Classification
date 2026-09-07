"""Bước 0a — parse 555 tên lá NABirds -> `zeroshot/data/leaf_variants.csv`.

Chạy: `python zeroshot/src/build_variants.py`

Verify (in ra cuối, phải đạt hết thì mới sang bước sau):
  - 555/555 lá parse được
  - 0 lá có ngoặc đơn mà `known=False`
  - tên gốc của lá trùng tên species cha (giả định mà mọi prompt dựa vào)
  - dựng lại được split unseen 'biến thể' (81 lá female/immature)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

import zs_env  # noqa: F401,E402  (đặt env trước mọi import nặng)
import nabirds_io as nio  # noqa: E402
import variants as V  # noqa: E402

OUT = os.path.join(zs_env.DATA_DIR, 'leaf_variants.csv')


def main():
    class_names = nio.load_class_names()
    hierarchy = nio.load_hierarchy()
    labels = nio.load_image_labels()
    train_ids, test_ids = nio.load_train_test_split()
    _, leaf_ids = nio.build_label_index(labels)
    taxonomy = nio.build_taxonomy(leaf_ids, hierarchy, class_names)

    n_train = pd.Series([labels[i] for i in train_ids]).value_counts()
    n_test = pd.Series([labels[i] for i in test_ids]).value_counts()

    rows, unknown, name_mismatch = [], [], []
    for cid in leaf_ids:
        leaf_name = class_names[cid]
        base, raw = V.split_leaf_name(leaf_name)
        rec = V.parse_variant(raw) if raw else None
        species_name, order_name = taxonomy[cid]

        if raw and not rec['known']:
            unknown.append((cid, leaf_name))
        if base != species_name:
            name_mismatch.append((leaf_name, species_name))

        rows.append({
            'class_id': cid,
            'leaf_name': leaf_name,
            'base_name': base,
            'species_name': species_name,
            'order_name': order_name,
            'variant_raw': raw or '',
            'phrase': rec['phrase'] if rec else '',
            'sex': '/'.join(rec['sex']) if rec else '',
            'age': '/'.join(rec['age']) if rec else '',
            'season': '/'.join(rec['season']) if rec else '',
            'morph': '/'.join(rec['morph']) if rec else '',
            'form': '/'.join(rec['form']) if rec else '',
            'female_side': bool(rec and V.is_female_side(rec)),
            'male_only': bool(rec and V.is_male_only(rec)),
            'n_train': int(n_train.get(cid, 0)),
            'n_test': int(n_test.get(cid, 0)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding='utf-8')

    # ---- bảng 60 chuỗi biến thể, để soi bằng mắt ----
    seen = df[df.variant_raw != ''].drop_duplicates('variant_raw')
    print(f'{"variant_raw":34s} {"-> phrase":40s} sex/age/season/morph/form')
    print('-' * 118)
    for _, r in seen.sort_values('variant_raw').iterrows():
        axes = '  '.join(x for x in (r.sex, r.age, r.season, r.morph, r.form) if x)
        print(f'{r.variant_raw:34s} -> {r.phrase:40s} {axes}')

    # ---- verify ----
    n_multi = df.groupby('species_name').size()
    male_sp = set(df[df.male_only].species_name)
    unseen = df[df.female_side & df.species_name.isin(male_sp)]

    print(f'\n{"="*60}\nVERIFY')
    print(f'  la                       : {len(df)} (can 555)')
    print(f'  la co bien the           : {(df.variant_raw != "").sum()} (can 288)')
    print(f'  chuoi bien the khac nhau : {seen.variant_raw.nunique()} (can 60)')
    print(f'  la KHONG parse duoc      : {len(unknown)} (can 0)')
    print(f'  base_name != species     : {len(name_mismatch)} (can 0)')
    print(f'  species                  : {df.species_name.nunique()} (can 404)')
    print(f'  species da-la            : {(n_multi > 1).sum()} (can 137)')
    seen_mates = df[df.male_only & df.species_name.isin(set(unseen.species_name))]
    print(f'  SPLIT bien the -> unseen : {len(unseen)} la, {unseen.n_test.sum()} anh test')
    print(f'    la male tuong ung (seen): {len(seen_mates)} la, '
          f'{seen_mates.n_train.sum()} anh train')
    print(f'    anh train BI BO khi giu lai {len(unseen)} la unseen: {unseen.n_train.sum()}')
    if unknown:
        print('  !! khong parse duoc:', unknown)
    if name_mismatch:
        print('  !! lech ten (5 vd):', name_mismatch[:5])
    print(f'\n-> {OUT}')


if __name__ == '__main__':
    main()
