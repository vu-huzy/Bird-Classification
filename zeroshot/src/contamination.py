"""Bước 1 — đối chiếu 404 loài NABirds với danh sách taxa mà BioCLIP đã embed.

Nguồn: `imageomics/TreeOfLife-10M`, file `embeddings/txt_emb_species.json` (66 MB).
Cấu trúc: list của `[[kingdom, phylum, class, order, family, genus, epithet], common_name]`
— đúng tập taxa mà BioCLIP dựng text embedding lúc pretrain.

Kết quả quyết định cách gọi tên phần đánh giá trong báo cáo: "zero-shot" thật sự
hay "đánh giá in-domain đã pretrain".

Chạy: `python zeroshot/src/contamination.py`
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zs_env  # noqa: E402

OUT = os.path.join(zs_env.RESULTS_DIR, 'contamination.csv')


def load_tol():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download('imageomics/TreeOfLife-10M', 'embeddings/txt_emb_species.json',
                        repo_type='dataset', cache_dir=os.path.join(zs_env.CACHE_DIR, 'hf'))
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def main():
    tx = pd.read_csv(os.path.join(zs_env.DATA_DIR, 'species_taxonomy.csv'),
                     keep_default_na=False)
    entries = load_tol()
    print(f'TreeOfLife-10M: {len(entries):,} taxa da embed')

    binomials, genera, families, commons = set(), set(), set(), set()
    common2bino = {}                                 # tên thường -> nhị thức, chỉ Aves
    n_aves = 0
    for tax, common in entries:
        if len(tax) < 7:
            continue
        _, _, cls, _, fam, gen, epi = tax[:7]
        bino = f'{gen} {epi}'.lower() if gen and epi else ''
        if bino:
            binomials.add(bino)
        if gen:
            genera.add(gen.lower())
        if fam:
            families.add(fam.lower())
        if common:
            commons.add(common.strip().lower())
        if cls == 'Aves':
            n_aves += 1
            if common and bino:
                common2bino.setdefault(common.strip().lower(), bino)
    print(f'  trong do class=Aves      : {n_aves:,} taxa')
    print(f'  ten nhi thuc duy nhat    : {len(binomials):,}')

    tx['in_tol_species'] = tx.scientific_name.str.lower().isin(binomials)
    tx['in_tol_genus'] = tx.genus.str.lower().isin(genera)
    tx['in_tol_family'] = tx.family.str.lower().isin(families)
    tx['in_tol_common'] = tx.common_name.str.lower().isin(commons)
    tx.to_csv(OUT, index=False, encoding='utf-8')

    n = len(tx)
    print(f'\n{"="*64}\nDOI CHIEU 404 LOAI NABirds')
    for col, label in [('in_tol_species', 'trung o muc LOAI (ten nhi thuc)'),
                       ('in_tol_genus', 'trung o muc CHI (genus)'),
                       ('in_tol_family', 'trung o muc HO (family)'),
                       ('in_tol_common', 'trung ten thuong')]:
        k = int(tx[col].sum())
        print(f'  {label:34s}: {k:3d}/{n} = {100*k/n:5.1f}%')

    miss = tx[~tx.in_tol_species]
    print(f'\n{len(miss)} loai KHONG trung o muc loai:')
    for _, r in miss.iterrows():
        flags = []
        if r.in_tol_genus:
            flags.append('genus CO')
        if r.in_tol_family:
            flags.append('family CO')
        print(f'  {r.common_name:34s} {r.scientific_name:28s} {" / ".join(flags) or "-"}')

    # ---- kiểm chéo độc lập: GBIF (nguồn của ta) vs ToL-10M (nguồn khác hẳn) ----
    both = tx[tx.common_name.str.lower().isin(common2bino)]
    agree = sum(common2bino[r.common_name.lower()] == r.scientific_name.lower()
                for _, r in both.iterrows())
    print(f'\n{"="*64}\nKIEM CHEO GBIF vs ToL-10M (2 nguon doc lap)')
    print(f'  loai co ten thuong o ca 2 : {len(both)}')
    print(f'  nhi thuc TRUNG KHOP       : {agree}/{len(both)} = {100*agree/len(both):.1f}%')
    disagree = [(r.common_name, r.scientific_name, common2bino[r.common_name.lower()])
                for _, r in both.iterrows()
                if common2bino[r.common_name.lower()] != r.scientific_name.lower()]
    for cn, mine, theirs in disagree:
        print(f'    {cn:32s} GBIF={mine:26s} ToL={theirs}')

    print(f'\n{"="*64}\nKET LUAN')
    frac = tx.in_tol_species.mean()
    if frac > 0.9:
        print(f'  {100*frac:.1f}% loai da nam trong pretrain cua BioCLIP.')
        print('  -> KHONG duoc goi ket qua tren NABirds la "zero-shot muc loai".')
        print('  -> Truc zero-shot hop le con lai: BIEN THE bo long (sex/tuoi/mua/morph),')
        print('     vi BioCLIP chi duoc giam sat bang taxonomy, khong co nhan bien the.')
    else:
        print(f'  Chi {100*frac:.1f}% trung -> co the noi zero-shot muc loai, nhung phai')
        print('  bao cao rieng nhom trung va nhom khong trung.')
    print(f'\n-> {OUT}')


if __name__ == '__main__':
    main()
