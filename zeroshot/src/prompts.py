"""Dựng chuỗi text cho từng lớp, ở 4 mức T0..T3.

Giao thức đánh giá bám đúng `examples/zero_shot.py` của BioCLIP và `predict.py` của
`pybioclip`: mỗi lớp được đưa qua **80 template ImageNet của OpenAI**, lấy trung bình
embedding đã chuẩn hoá rồi chuẩn hoá lại. Không phải một câu duy nhất.

Nguồn taxonomy có HAI bản, chọn bằng `taxo=`:
  `tol`  — chuỗi ToL-10M, tức chuỗi BioCLIP **thật sự được train** (`Dendroica coronata`)
  `gbif` — taxonomy hiện hành, dùng cho CLIP gốc và để báo cáo (`Setophaga coronata`)
33/404 loài khác nhau giữa hai bản.

Bốn mức text (biến duy nhất trong thí nghiệm bước 2):
  T0  tên thường của loài                       -> KHÔNG tách được 288 lá biến thể
  T1  chuỗi taxonomy đầy đủ + tên thường        -> vẫn KHÔNG tách được (taxonomy dừng ở loài)
  T2  T1 + cụm biến thể đã chuẩn hoá            -> tách được
  T3  T2 + mô tả hình thái (bước sau, có điều kiện)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zs_env  # noqa: E402
import descriptors  # noqa: E402

# 80 template ImageNet của OpenAI — chép đúng thứ tự trong `pybioclip/predict.py`.
OPENAI_TEMPLATES = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.',
    'a photo of the hard to see {}.', 'a low resolution photo of the {}.',
    'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.',
    'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.',
    'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.',
    'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.',
    'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.',
    'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
    'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.',
    'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.',
    'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.',
    'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.',
    'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.',
    'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.',
    'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.',
    'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.',
    'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.',
    'a pixelated photo of a {}.', 'itap of the {}.',
    'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.',
    'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.',
    'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.',
    'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.',
    'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.',
    'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.',
    'a tattoo of the {}.',
]
assert len(OPENAI_TEMPLATES) == 80

# Thiết kế 2 x 3: (có/không chuỗi taxonomy) x (không biến thể / biến thể thô / +mô tả).
# T3a làm SỤT kết quả so với T2 -> nghi mô tả dài đẩy embedding ra khỏi vùng taxonomy
# mà BioCLIP quen. Nhánh không-taxonomy (T0v/T0d) là phép đối chứng cho nghi vấn đó.
#                  taxonomy  biến thể  mô tả
SPEC = {
    'T0':  (False, False, None),        # chỉ tên thường
    'T0v': (False, True,  None),        # tên thường + cụm biến thể
    'T0d': (False, True,  'generic'),   # + mô tả chung
    'T0s': (False, True,  'specific'),  # + mô tả riêng loài
    'T1':  (True,  False, None),        # taxonomy + tên thường  (format gốc BioCLIP)
    'T2':  (True,  True,  None),
    'T3a': (True,  True,  'generic'),
    'T3b': (True,  True,  'specific'),
    # Doi chung: y het T0s/T3b nhung mo ta bi hoan vi giua cac loai (giu vai tro
    # male/female). Dung de tach "noi dung dung" khoi "chi can text da dang".
    'T0sx': (False, True,  'shuffled'),
    'T3bx': (True,  True,  'shuffled'),
}
LEVELS = tuple(SPEC)


def load_tables():
    """Gộp 3 file của bước 0/1 -> 1 bảng 555 dòng, đủ để dựng mọi mức prompt."""
    d = zs_env.DATA_DIR
    leaf = pd.read_csv(os.path.join(d, 'leaf_variants.csv'), keep_default_na=False)
    gbif = pd.read_csv(os.path.join(d, 'species_taxonomy.csv'), keep_default_na=False)
    tol = pd.read_csv(os.path.join(d, 'tol_taxonomy.csv'), keep_default_na=False)

    gbif = gbif.rename(columns={c: f'gbif_{c}' for c in
                                ('kingdom', 'phylum', 'class', 'order', 'family',
                                 'genus', 'scientific_name')})
    df = (leaf.merge(gbif, left_on='species_name', right_on='common_name', how='left')
              .merge(tol.drop(columns=['common_name', 'gbif_scientific']),
                     left_on='species_class_id', right_on='species_class_id', how='left'))
    assert len(df) == 555 and df.tol_genus.ne('').all() and df.gbif_genus.ne('').all()
    return df


def _taxon_string(r, taxo):
    """Chuỗi lớp theo đúng `join_names` của pybioclip:
    `kingdom phylum class order family genus epithet genus epithet common_name`
    (nhị thức xuất hiện hai lần — là hành vi thật của thư viện, không phải lỗi chép)."""
    if taxo == 'tol':
        ranks = [r.tol_kingdom, r.tol_phylum, r.tol_class, r.tol_order,
                 r.tol_family, r.tol_genus, r.tol_epithet]
        bino = f'{r.tol_genus} {r.tol_epithet}'
    else:
        gen, epi = r.gbif_scientific_name.split()
        ranks = [r.gbif_kingdom, r.gbif_phylum, r.gbif_class, r.gbif_order,
                 r.gbif_family, gen, epi]
        bino = r.gbif_scientific_name
    return ' '.join(ranks + [bino])


def class_texts(df, level, space='leaf', taxo='tol'):
    """Trả về (danh sách chuỗi lớp, danh sách tên lớp) theo thứ tự nhãn 0..N-1.

    `space='leaf'`    -> 555 lớp, thứ tự theo `class_id` tăng dần (khớp `build_label_index`)
    `space='species'` -> 404 lớp, thứ tự theo `species_class_id` tăng dần
    """
    if space == 'species':
        sub = df.drop_duplicates('species_class_id').sort_values(
            'species_class_id', key=lambda s: s.astype(int))
    else:
        sub = df.sort_values('class_id', key=lambda s: s.astype(int))

    use_taxo, use_variant, desc = SPEC[level]
    specific = _load_specific(desc) if desc in ('specific', 'shuffled') else {}

    texts, names = [], []
    for _, r in sub.iterrows():
        common = r.species_name
        t = f'{_taxon_string(r, taxo)} {common}' if use_taxo else common

        # Biến thể chỉ có nghĩa ở không gian nhãn lá
        if use_variant and space == 'leaf' and r.phrase:
            t = f'{t} {r.phrase}'
            generic = descriptors.generic_phrase(
                _split(r.sex), _split(r.age), _split(r.season),
                _split(r.morph), _split(r.form))
            if desc == 'generic':
                t += generic
            elif desc in ('specific', 'shuffled'):
                t += specific.get(r.leaf_name) or generic   # thiếu thì lùi về mô tả chung
        texts.append(t)
        names.append(r.leaf_name if space == 'leaf' else common)
    return texts, names


def _split(v):
    """Cột CSV 'female/male' -> ('female', 'male'); rỗng -> ()."""
    return tuple(x for x in str(v).split('/') if x)


def _load_specific(kind='specific'):
    """Mô tả riêng từng lá (file do `build_descriptors.py` sinh).

    `kind='shuffled'` lấy bản đã hoán vị giữa các loài — phép đối chứng.
    """
    import json
    fn = ('leaf_descriptors.json' if kind == 'specific'
          else 'leaf_descriptors_shuffled.json')
    p = os.path.join(zs_env.DATA_DIR, fn)
    if not os.path.exists(p):
        return {}
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def expand(texts):
    """1 chuỗi lớp -> 80 câu prompt. Trả về list phẳng + số câu mỗi lớp."""
    return [tpl.format(t) for t in texts for tpl in OPENAI_TEMPLATES], len(OPENAI_TEMPLATES)
