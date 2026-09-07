"""Bước 0b — ánh xạ 404 common name NABirds -> taxonomy Latin, qua GBIF Backbone.

Vì sao cần: NABirds **không có tên khoa học** ở bất kỳ file nào. `classes.txt` ghi
`Yellow-rumped Warbler` và order là `Perching Birds`, trong khi prompt gốc của
BioCLIP cần `Aves Passeriformes Parulidae Setophaga coronata`. Bảng này cũng là
điều kiện để đối chiếu contamination với TreeOfLife-10M (danh sách của họ bằng
tên Latin).

Nguồn: GBIF Backbone Taxonomy (API mở, không cần key). Lọc `class == Aves` vì
tra theo tên thường dễ trúng côn trùng/ve ký sinh trùng tên với chim.

Chạy: `python zeroshot/src/build_taxonomy.py`
Kết quả cache theo từng loài ở `zeroshot/cache/gbif/` -> chạy lại gần như tức thì.
"""
import json
import os
import re
import sys
import time
import urllib.parse

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

import zs_env  # noqa: F401,E402
import nabirds_io as nio  # noqa: E402

BACKBONE = 'd7dddbf4-2cf0-4f39-9b2a-bb099caae36c'   # GBIF Backbone Taxonomy
API = 'https://api.gbif.org/v1/species/search'
CACHE = os.path.join(zs_env.CACHE_DIR, 'gbif')
OUT = os.path.join(zs_env.DATA_DIR, 'species_taxonomy.csv')
RANKS = ('kingdom', 'phylum', 'class', 'order', 'family', 'genus')

# Tên thường tra thẳng bị hỏng -> tra bằng chuỗi khác. Việc xác minh VẪN so với
# tên NABirds gốc trong danh sách vernacular của GBIF, nên alias không nới lỏng
# tiêu chuẩn khớp.
#   Redhead: tra "Redhead" trả về toàn nấm (Agaricomycetes) vì đây là một từ đơn
#            phổ thông; vernacular của Aythya americana có đúng "Redhead".
#
# Ba loài dưới đây: NABirds đóng băng ở taxonomy 2015, GBIF đã cập nhật. Tra bằng
# tên thường sẽ đổ dồn hai loài NABirds vào MỘT bản ghi GBIF (đụng độ tên khoa
# học). Tra thẳng bằng tên nhị thức bản 2015 để giữ đúng ranh giới lớp của dataset:
#   Hoary Redpoll         : AOS 2024 gộp vào Common Redpoll (Acanthis flammea)
#   Cordilleran Flycatcher: AOS 2023 gộp vào Pacific-slope (Empidonax difficilis)
#   Veery                 : không bị gộp — GBIF liệt "Veery" nhầm vào vernacular
#                           của Catharus ustulatus (Swainson's Thrush)
#
# Nhóm cuối do bước kiểm chéo với ToL-10M phát hiện (README 10.2). NABirds là chim
# BẮC MỸ, nhưng luật "nhiều vernacular nhất" hay chọn trúng loài Cựu Thế giới trùng
# tên tiếng Anh, hoặc loài gộp mà bản Tân Thế giới đã được tách ra.
ALIAS = {
    'Redhead': 'Aythya americana',
    # taxonomy đổi sau 2015 -> tra tên thường làm 2 lớp NABirds đụng nhau
    'Hoary Redpoll': 'Acanthis hornemanni',
    'Cordilleran Flycatcher': 'Empidonax occidentalis',
    'Veery': 'Catharus fuscescens',
    # GBIF trả loài Cựu Thế giới khác hẳn
    'Anhinga': 'Anhinga anhinga',                    # không phải Oriental Darter
    'Black Vulture': 'Coragyps atratus',             # không phải Cinereous Vulture
    'Purple Gallinule': 'Porphyrio martinica',       # không phải Western Swamphen
    # tách loài Tân/Cựu Thế giới — NABirds dùng bản Bắc Mỹ
    'Black-billed Magpie': 'Pica hudsonia',
    'Green-winged Teal': 'Anas carolinensis',
    'Northern Shrike': 'Lanius borealis',
    'Winter Wren': 'Troglodytes hiemalis',
    'White-winged Scoter': 'Melanitta deglandi',
    'Mexican Jay': 'Aphelocoma wollweberi',
}
# Ba loài KHÔNG sửa được: GBIF xếp `Gallinula galeata` (Common Gallinule),
# `Circus hudsonius` (Northern Harrier), `Caracara cheriway` (Crested Caracara) là
# SYNONYM của loài gộp, không có bản ghi ACCEPTED. Giữ theo GBIF; prompt cho BioCLIP
# dùng chuỗi taxonomy của ToL-10M nên không ảnh hưởng.


def _norm(s):
    """Chuẩn hoá để so tên thường: bỏ dấu nháy, gạch nối, hoa/thường, khoảng trắng."""
    return re.sub(r'[^a-z ]', '', s.lower().replace('-', ' ')).strip()


def gbif_search(name):
    """Tra 1 tên thường, cache ra file JSON. Trả về list kết quả GBIF."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, re.sub(r'[^A-Za-z0-9]+', '_', name) + '.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    url = (f'{API}?q={urllib.parse.quote(name)}&rank=SPECIES&status=ACCEPTED'
           f'&datasetKey={BACKBONE}&limit=20')
    for attempt in range(3):
        try:
            res = requests.get(url, timeout=30).json().get('results', [])
            break
        except Exception as e:                       # mạng chập chờn -> thử lại
            if attempt == 2:
                raise
            print(f'   retry {name}: {e}')
            time.sleep(2)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f)
    time.sleep(0.15)                                 # lịch sự với API công cộng
    return res


def _binomial(r):
    """Bản ghi có tên nhị thức hợp lệ không?

    Lọc bản ghi LAI: GBIF trả `Anas rubripes x platyrhynchos` (Black Duck x Mallard)
    cho tên thường "American Black Duck", và bản ghi lai KHÔNG có `canonicalName`.
    Không lọc thì loài đó ra rỗng dù vẫn báo là khớp.
    """
    cn = r.get('canonicalName') or ''
    return len(cn.split()) == 2


def pick(name, results):
    """Chọn bản ghi Aves có tên thường khớp. Trả về (record, cách khớp).

    Hai luật lọc, cả hai đều do bug thật ép ra (xem README 10.2):

    1. Bỏ loài HOÁ THẠCH. GBIF chứa `Aquila bivia`, `Pandion lovensis`,
       `Tachycineta speleodytes` — đều `extinct=True` và mang đúng tên thường
       "Golden Eagle" / "Osprey" / "Tree Swallow" của loài còn sống.
    2. Trong số các bản ghi khớp, lấy bản có NHIỀU tên thường nhất. Loài phổ biến
       có 60–288 vernacular, bản ghi hoá thạch chỉ có 1. Luật này bắt nốt
       `Icterus bullockiorum` (extinct=None nên luật 1 không bắt được).
    """
    aves = [r for r in results
            if r.get('class') == 'Aves' and r.get('genus') and _binomial(r)
            and not r.get('extinct')]
    if not aves:
        return None, 'no_aves'
    target = _norm(name)
    hits = [r for r in aves
            if target in {_norm(v.get('vernacularName', ''))
                          for v in r.get('vernacularNames', [])}]
    if hits:
        return max(hits, key=lambda r: len(r.get('vernacularNames', []))), 'vernacular_exact'
    return aves[0], 'aves_first'                     # cần soi tay


def main():
    class_names = nio.load_class_names()
    hierarchy = nio.load_hierarchy()
    labels = nio.load_image_labels()
    _, leaf_ids = nio.build_label_index(labels)

    species = {}                                     # species_class_id -> ten
    for cid in leaf_ids:
        species[hierarchy[cid]] = class_names[hierarchy[cid]]
    print(f'{len(species)} species can tra\n')

    rows = []
    for n, (sid, name) in enumerate(sorted(species.items(), key=lambda kv: kv[1]), 1):
        query = ALIAS.get(name, name)
        rec, how = pick(name, gbif_search(query))    # xác minh luôn theo tên NABirds
        rows.append({
            'species_class_id': sid,
            'common_name': name,
            'query_used': query,
            'scientific_name': rec.get('canonicalName') if rec else '',
            **{r: (rec.get(r, '') if rec else '') for r in RANKS},
            'match': how,
            'n_vernacular': len(rec.get('vernacularNames', [])) if rec else 0,
        })
        if n % 50 == 0:
            print(f'  {n}/{len(species)}')

    df = pd.DataFrame(rows).sort_values('common_name')
    df.to_csv(OUT, index=False, encoding='utf-8')

    ok = (df.match == 'vernacular_exact').sum()
    n_binom = (df.scientific_name.str.split().str.len() == 2).sum()
    print(f'\n{"="*60}\nVERIFY')
    print(f'  khop chinh xac ten thuong : {ok}/{len(df)} = {100*ok/len(df):.1f}%  (can >=95%)')
    print(f'  ten nhi thuc hop le       : {n_binom}/{len(df)}  (can bang nhau)')
    print(f'  ten khoa hoc duy nhat     : {df.scientific_name.nunique()}/{len(df)}  (can bang nhau)')
    print(f'  chi lay Aves dau tien     : {(df.match == "aves_first").sum()}  (phai soi tay)')
    thin = df[df.n_vernacular <= 2]
    print(f'  ban ghi it vernacular (<=2): {len(thin)}  (nghi ngo hoa thach/loi, can 0)')
    for _, r in thin.iterrows():
        print(f'      {r.common_name} -> {r.scientific_name} (nVern={r.n_vernacular})')
    print(f'  khong tim thay Aves       : {(df.match == "no_aves").sum()}')
    print(f'  order Latin khac nhau     : {df[df["order"] != ""]["order"].nunique()}'
          f'  (NABirds gop thanh 22 "order" tieng Anh)')

    bad = df[df.match != 'vernacular_exact']
    if len(bad):
        print(f'\n--- {len(bad)} dong can xu ly tay ---')
        for _, r in bad.iterrows():
            print(f'  {r.common_name:36s} -> {r.scientific_name or "(khong co)":32s} [{r.match}]')
    print(f'\n-> {OUT}')


if __name__ == '__main__':
    main()
