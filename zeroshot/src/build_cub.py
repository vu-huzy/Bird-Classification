"""Bước 3c (B4) — chuẩn bị CUB-200-2011 + caption, và split unseen CUB -> NABirds.

Vì sao cần CUB: mục 3.2 chỉ ra NABirds chỉ có text MỨC LỚP, nên phương pháp thực chất
là DeViSE/ALE chứ không phải contrastive kiểu CLIP/METS. METS ghép mỗi ECG với MỘT báo
cáo riêng (máy sinh). Muốn tái hiện đúng cấu trúc đó thì cần text biến thiên theo từng
ảnh — CUB có, NABirds không.

Hai bộ caption trên cùng 5,994 ảnh CUB train, khác nhau ĐÚNG một biến là độ hạt của text:

  `naive` — `anjunhu/naively_captioned_CUB2002011_train`
            "A photo of a laysan albatross" -> **200 chuỗi duy nhất / 5,994 ảnh** (mức LỚP)
  `cupl`  — `anjunhu/CuPL_DaVinci_captioned_CUB2002011_train`
            mô tả hình thái do LLM sinh -> **4,490 chuỗi duy nhất / 5,994 ảnh** (biến thiên)

Đây là phép đo trực tiếp cho luận điểm ở mục 3.2. Nói thẳng giới hạn: caption CuPL sinh
từ TÊN LỚP chứ không phải từ chính bức ảnh, nên đây là "text mức lớp có biến thiên",
chưa phải text thật sự điều kiện theo ảnh như caption người viết của Reed et al. 2016
(bộ đó không có bản tải tự động được).

Chạy: `python zeroshot/src/build_cub.py`
"""
import io
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

import zs_env  # noqa: E402

REPOS = {'naive': 'anjunhu/naively_captioned_CUB2002011_train',
         'cupl': 'anjunhu/CuPL_DaVinci_captioned_CUB2002011_train'}

# Tên CUB không khớp thẳng tên NABirds. Danh sách này đã ghi ở mục 0 của README.
ALIAS = {
    'cardinal': 'Northern Cardinal', 'mockingbird': 'Northern Mockingbird',
    'white pelican': 'American White Pelican', 'tree sparrow': 'American Tree Sparrow',
    'florida jay': 'Florida Scrub-Jay', 'forsters tern': "Forster's Tern",
    'nighthawk': 'Common Nighthawk', 'myrtle warbler': 'Yellow-rumped Warbler',
    'geococcyx': 'Greater Roadrunner', 'great grey shrike': 'Northern Shrike',
    'sayornis': 'Eastern Phoebe', 'whip poor will': 'Eastern Whip-poor-will',
    'chuck will widow': "Chuck-will's-widow", 'brandt cormorant': "Brandt's Cormorant",
    'pelagic cormorant': 'Pelagic Cormorant', 'gray crowned rosy finch':
        'Gray-crowned Rosy-Finch', 'pigeon guillemot': 'Pigeon Guillemot',
}


def _norm(s):
    """Chuẩn hoá tên loài để so khớp.

    Bỏ đuôi sở hữu cách TRƯỚC khi bỏ ký tự lạ: CUB ghi `wilson warbler`, NABirds ghi
    `Wilson's Warbler`. Nếu chỉ thay dấu nháy bằng khoảng trắng thì thành
    `wilson s warbler` -> lệch một token và trượt hàng loạt loài phổ biến.
    """
    s = s.lower().replace('’', "'")
    s = re.sub(r"'s\b", '', s).replace("'", '')
    return re.sub(r'[^a-z ]', ' ', s.replace('-', ' ')).split()


def load_split(tag):
    """-> DataFrame(text, image_bytes). Ảnh nằm ngay trong parquet."""
    from huggingface_hub import hf_hub_download, list_repo_files
    repo = REPOS[tag]
    fs = [f for f in list_repo_files(repo, repo_type='dataset') if f.endswith('.parquet')]
    d = pd.read_parquet(hf_hub_download(repo, fs[0], repo_type='dataset',
                                        cache_dir=os.path.join(zs_env.CACHE_DIR, 'hf')))
    d['image_bytes'] = [x['bytes'] for x in d.image]
    return d[['text', 'image_bytes']]


def cub_class_names(naive_df):
    """200 tên lớp CUB, rút từ caption 'A photo of a {ten}'."""
    return sorted(naive_df.text.str.replace(r'^A photo of a\s*', '', regex=True)
                  .str.strip().unique())


def match_to_nabirds(cub_names, species_names):
    """Ghép tên lớp CUB -> tên loài NABirds. Trả về dict + danh sách không khớp."""
    by_norm = {' '.join(_norm(s)): s for s in species_names}
    hit, miss = {}, []
    for c in cub_names:
        key = ' '.join(_norm(c))
        if key in ALIAS:
            hit[c] = ALIAS[key]
        elif key in by_norm:
            hit[c] = by_norm[key]
        else:                                   # khớp theo tập token (thứ tự khác nhau)
            toks = set(_norm(c))
            cands = [s for k, s in by_norm.items() if set(k.split()) == toks]
            if len(cands) == 1:
                hit[c] = cands[0]
            else:
                miss.append(c)
    return hit, miss


def main():
    import prompts
    df = prompts.load_tables()
    sub = df.sort_values('class_id', key=lambda s: s.astype(int)).reset_index(drop=True)

    naive = load_split('naive')
    cub_names = cub_class_names(naive)
    hit, miss = match_to_nabirds(cub_names, sorted(sub.species_name.unique()))

    seen_species = set(hit.values())
    unseen_leaf = ~sub.species_name.isin(seen_species).to_numpy()
    np.save(os.path.join(zs_env.DATA_DIR, 'cub_unseen_leaf.npy'), unseen_leaf)
    pd.DataFrame({'cub_class': cub_names,
                  'nabirds_species': [hit.get(c, '') for c in cub_names]}).to_csv(
        os.path.join(zs_env.DATA_DIR, 'cub_to_nabirds.csv'), index=False,
        encoding='utf-8')

    y_te = np.load(os.path.join(zs_env.CACHE_DIR, 'emb', 'labels_test.npy'))
    n_te = np.bincount(y_te, minlength=555)
    print(f'{"="*64}\nCUB -> NABirds')
    print(f'  lop CUB khop duoc loai NABirds : {len(hit)}/200')
    print(f'  loai NABirds bi CUB phu (seen) : {len(seen_species)}/404')
    print(f'  la NABirds UNSEEN              : {int(unseen_leaf.sum())}/555')
    print(f'  anh test thuoc la unseen       : {int(n_te[unseen_leaf].sum())}/{len(y_te)}')
    if miss:
        print(f'\n  {len(miss)} lop CUB khong khop (loai bien / khong co o Bac My):')
        for m in miss:
            print(f'      {m}')

    for tag in REPOS:
        d = load_split(tag)
        print(f'\n{tag}: {len(d)} anh, {d.text.nunique()} chuoi text duy nhat '
              f'({d.text.nunique()/len(d):.2f} text/anh)')
        print('   vd:', d.text.iloc[0][:110])


if __name__ == '__main__':
    main()
