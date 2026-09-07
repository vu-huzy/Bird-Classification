"""Bước 1b — lấy chuỗi taxonomy ĐÚNG NHƯ BioCLIP đã thấy, cho 404 loài NABirds.

Vì sao cần file này bên cạnh `species_taxonomy.csv` (GBIF): bước kiểm chéo cho thấy
ToL-10M dùng taxonomy **cũ hơn** GBIF ở 33/372 loài — `Dendroica coronata` chứ không
phải `Setophaga coronata`, `Carduelis pinus` chứ không phải `Spinus pinus`,
`Parus bicolor` chứ không phải `Baeolophus bicolor`...

Prompt T1 phải dùng chuỗi mà model **thật sự được train**, nếu không thì đang đo
"BioCLIP kém" trong khi thực chất là đưa cho nó chuỗi ngoài phân phối.

  - `tol_*`  -> dùng cho prompt của BioCLIP / BioCLIP-2
  - `gbif_*` -> taxonomy hiện hành, dùng để báo cáo và cho prompt của CLIP gốc

Chạy: `python zeroshot/src/build_tol_taxonomy.py`
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zs_env  # noqa: E402

OUT = os.path.join(zs_env.DATA_DIR, 'tol_taxonomy.csv')
RANKS = ('kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'epithet')


def load_tol_aves():
    """Chỉ lấy nhánh Aves của ToL-10M -> (theo tên thường, theo nhị thức)."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download('imageomics/TreeOfLife-10M', 'embeddings/txt_emb_species.json',
                        repo_type='dataset', cache_dir=os.path.join(zs_env.CACHE_DIR, 'hf'))
    with open(p, encoding='utf-8') as f:
        entries = json.load(f)

    by_common, by_bino = {}, {}
    for tax, common in entries:
        if len(tax) < 7 or tax[2] != 'Aves':
            continue
        rec = dict(zip(RANKS, tax[:7]), common=common.strip())
        if common:
            by_common.setdefault(common.strip().lower(), rec)
        if tax[5] and tax[6]:
            by_bino.setdefault(f'{tax[5]} {tax[6]}'.lower(), rec)
    return by_common, by_bino


def main():
    gbif = pd.read_csv(os.path.join(zs_env.DATA_DIR, 'species_taxonomy.csv'),
                       keep_default_na=False)
    by_common, by_bino = load_tol_aves()
    print(f'ToL-10M nhanh Aves: {len(by_bino):,} nhi thuc, {len(by_common):,} ten thuong')

    rows = []
    for _, r in gbif.iterrows():
        rec = by_common.get(r.common_name.lower())
        how = 'common_name'
        if rec is None:
            rec = by_bino.get(r.scientific_name.lower())
            how = 'binomial' if rec else 'none'
        rows.append({
            'species_class_id': r.species_class_id,
            'common_name': r.common_name,
            'gbif_scientific': r.scientific_name,
            **{f'tol_{k}': (rec[k] if rec else '') for k in RANKS},
            'tol_common': rec['common'] if rec else '',
            'tol_match': how,
        })

    df = pd.DataFrame(rows)
    df['tol_scientific'] = (df.tol_genus + ' ' + df.tol_epithet).str.strip()
    df.to_csv(OUT, index=False, encoding='utf-8')

    n = len(df)
    print(f'\n{"="*64}\nVERIFY')
    for how in ('common_name', 'binomial', 'none'):
        k = int((df.tol_match == how).sum())
        print(f'  khop bang {how:12s}: {k:3d}/{n}')
    same = int((df.tol_scientific.str.lower() == df.gbif_scientific.str.lower()).sum())
    print(f'  ToL == GBIF (nhi thuc) : {same}/{n} = {100*same/n:.1f}%')
    print(f'  -> {n - same} loai se co chuoi T1 KHAC nhau giua BioCLIP va CLIP')
    for c in ('tol_order', 'tol_family', 'tol_genus'):
        print(f'  {c} rong             : {int((df[c] == "").sum())}')

    miss = df[df.tol_match == 'none']
    if len(miss):
        print(f'\n{len(miss)} loai khong co trong ToL-10M (dung taxonomy GBIF thay the):')
        for _, r in miss.iterrows():
            print(f'  {r.common_name:32s} {r.gbif_scientific}')
    print(f'\n-> {OUT}')


if __name__ == '__main__':
    main()
