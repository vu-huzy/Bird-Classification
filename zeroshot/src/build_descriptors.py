"""T3b — mô tả hình thái RIÊNG TỪNG LÁ cho 81 loài của split `variant`.

Vì sao làm bước này (có căn cứ đo, không phải đoán): ở bước 2, thêm mô tả thị giác
CHUNG (T0d) nâng oracle-variant của CLIP từ 64.98 -> 66.37, nhưng lại làm BioCLIP tệ đi.
Tức là mô tả bằng ngôn ngữ tự nhiên chỉ có tác dụng với text tower biết nối ngôn ngữ
với ảnh. Câu hỏi tiếp theo: nếu mô tả không còn chung chung mà nêu đúng đặc điểm nhận
dạng của TỪNG loài thì CLIP còn tiến thêm bao nhiêu?

Phạm vi: 81 loài của split `variant` (nơi đo con số zero-shot chính). 267 lá đơn và
các lá đa-biến-thể ngoài phạm vi này vẫn dùng mô tả chung của `descriptors.py`.

Nguồn: đặc điểm nhận dạng phổ thông của chim Bắc Mỹ. Mỗi cụm <= 12 từ vì tokenizer
CLIP chỉ có 77 token. CẢNH BÁO TRUNG THỰC: đây là văn bản do LLM sinh, cần soi tay
trước khi trích dẫn — xem `--check` để in ra toàn bộ và đối chiếu.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zs_env  # noqa: E402

# species -> {chuỗi trong ngoặc đơn: cụm mô tả}
D = {
 'Common Eider': {'Adult male': 'white above, black below, black cap, pale green nape',
                  'Female/juvenile': 'entirely warm brown with dense black barring',
                  'Immature/Eclipse male': 'blotchy brown and white, dark with irregular white patches'},
 'Wood Duck': {'Breeding male': 'iridescent green swept-back crest, white throat stripes, chestnut breast',
               'Female/Eclipse male': 'gray-brown with a white teardrop patch around the eye'},
 'Gadwall': {'Breeding male': 'plain gray body, black rear end, brown head',
             'Female/Eclipse male': 'mottled brown with an orange-edged dark bill'},
 'American Wigeon': {'Breeding male': 'green eye stripe and creamy white forehead stripe',
                     'Female/Eclipse male': 'gray head, warm brown body, small blue-gray bill'},
 'Mallard': {'Breeding male': 'glossy green head, yellow bill, white neck ring, gray body',
             'Female/Eclipse male': 'mottled brown with an orange and black bill'},
 'Blue-winged Teal': {'Male': 'gray head with a bold white crescent in front of the eye',
                      'Female/juvenile': 'plain mottled brown with a dark eye line'},
 'Cinnamon Teal': {'Male': 'entirely deep cinnamon-red with a red eye',
                   'Female/juvenile': 'plain mottled brown with a long spatulate bill'},
 'Northern Shoveler': {'Breeding male': 'green head, white breast, rusty flanks, huge spoon bill',
                       'Female/Eclipse male': 'mottled brown with an oversized orange spoon-shaped bill'},
 'Northern Pintail': {'Breeding male': 'chocolate head, white neck stripe, long pointed black tail',
                      'Female/Eclipse male': 'plain buff-brown, slender neck, pointed tail, gray bill'},
 'Green-winged Teal': {'Male': 'chestnut head with a green comma behind the eye',
                       'Female/juvenile': 'small mottled brown duck with a green wing patch'},
 'Canvasback': {'Breeding male': 'white body, rusty head, sloping black wedge-shaped bill',
                'Female/Eclipse male': 'pale gray body, tan head, same sloping bill profile'},
 'Redhead': {'Breeding male': 'round cinnamon-red head, gray body, black breast',
             'Female/Eclipse male': 'uniform warm brown with a pale eye ring and gray bill'},
 'Ring-necked Duck': {'Breeding male': 'black back, gray flanks, peaked head, white bill ring',
                      'Female/Eclipse male': 'brown with a peaked head, white eye ring, ringed bill'},
 'Greater Scaup': {'Breeding male': 'green-glossed round head, white flanks, pale back',
                   'Female/Eclipse male': 'brown with a large white patch at the bill base'},
 'Lesser Scaup': {'Breeding male': 'purple-glossed peaked head, white flanks, barred gray back',
                  'Female/Eclipse male': 'brown with a white patch at the bill base, peaked head'},
 'Harlequin Duck': {'Male': 'slate blue with bold white crescents and chestnut flanks',
                    'Female/juvenile': 'dusky brown with two or three round white face spots'},
 'Surf Scoter': {'Male': 'black with white forehead and nape patches, orange bill',
                 'Female/immature': 'brown with two pale patches on the side of the face'},
 'White-winged Scoter': {'Male': 'black with a white eye comma and a white wing patch',
                         'Female/juvenile': 'brown with two pale face patches and a white wing patch'},
 'Black Scoter': {'Male': 'entirely glossy black with a bulbous orange-yellow bill knob',
                  'Female/juvenile': 'brown with a pale cheek contrasting a dark cap'},
 'Bufflehead': {'Breeding male': 'small, white body, dark head with a big white wedge',
                'Female/immature male': 'small dusky gray-brown with a single white cheek stripe'},
 'Common Goldeneye': {'Breeding male': 'white body, green-black head, round white spot by the bill',
                      'Female/Eclipse male': 'gray body, chocolate brown head, yellow-tipped dark bill'},
 "Barrow's Goldeneye": {'Breeding male': 'purple-black head with a white crescent, blacker back',
                        'Female/Eclipse male': 'gray body, brown head, mostly yellow-orange bill'},
 'Hooded Merganser': {'Breeding male': 'huge fan-shaped white crest bordered black, rusty flanks',
                      'Female/immature male': 'gray-brown with a shaggy cinnamon crest, thin bill'},
 'Common Merganser': {'Breeding male': 'clean white body, dark green head, thin red bill',
                      'Female/immature male': 'gray body, rusty head with a shaggy crest, white chin'},
 'Red-breasted Merganser': {'Breeding male': 'spiky double crest, green head, white collar, rusty breast',
                            'Female/immature male': 'gray-brown with a shaggy rusty head, no sharp contrast'},
 'California Quail': {'Male': 'black face outlined white, forward-curling teardrop topknot',
                      'Female/juvenile': 'plain gray-brown face, smaller topknot, scaled belly'},
 "Gambel's Quail": {'Male': 'black face and belly patch, chestnut crown, curling plume',
                    'Female/juvenile': 'plain gray head without the black face, short plume'},
 'Ring-necked Pheasant': {'Male': 'iridescent green head, red face wattles, white neck ring',
                          'Female/juvenile': 'plain mottled buff-brown with a long pointed tail'},
 'Northern Harrier': {'Adult male': 'pale silver-gray above, white below, black wingtips',
                      'Female, immature': 'brown above, streaked warm buff below, white rump patch'},
 'American Kestrel': {'Adult male': 'blue-gray wings, rusty back and tail, black face stripes',
                      'Female, immature': 'rusty wings, back and tail all barred with black'},
 'Broad-billed Hummingbird': {'Adult Male': 'glittering green body, deep blue throat, red black-tipped bill',
                              'Female, immature': 'green above, plain gray below, white stripe behind eye'},
 'Ruby-throated Hummingbird': {'Adult Male': 'brilliant ruby-red throat patch, green back, black chin',
                               'Female, immature': 'green above, plain whitish throat and underparts'},
 'Black-chinned Hummingbird': {'Adult Male': 'black throat with a thin iridescent violet band below',
                               'Female, immature': 'green above, dull white below, plain pale throat'},
 "Anna's Hummingbird": {'Adult Male': 'entire head and throat glowing iridescent rose-pink',
                        'Female, immature': 'green above, gray below, small red throat flecks'},
 "Costa's Hummingbird": {'Adult Male': 'deep violet crown and long flaring violet throat feathers',
                         'Female, immature': 'green above, clean white below, plain face'},
 'Calliope Hummingbird': {'Adult Male': 'magenta throat rays streaked over a white background',
                          'Female, immature': 'tiny, green above, peachy buff wash on the flanks'},
 'Broad-tailed Hummingbird': {'Adult Male': 'rose-magenta throat, bright green back, white breast',
                              'Female, immature': 'green above, buffy flanks, speckled throat'},
 'Rufous Hummingbird': {'Adult Male': 'entirely bright orange-rufous with a fiery orange-red throat',
                        'Female, immature': 'green back, rufous flanks and tail base, spotted throat'},
 "Allen's Hummingbird": {'Adult Male': 'green back with rufous flanks and a coppery orange throat',
                         'Female, immature': 'green above, rufous wash on the flanks, spotted throat'},
 'Long-tailed Duck': {'Winter male': 'white head with a dark cheek patch, very long tail plumes',
                      'Summer male': 'dark chocolate head with a white face patch, long tail',
                      'Female/juvenile': 'brown and white with a plain round head, no long tail'},
 'Ruddy Duck': {'Breeding male': 'chestnut body, white cheek, black cap, sky-blue bill',
                'Winter male': 'dull gray-brown body, white cheek, dark cap, gray bill',
                'Female/juvenile': 'brown with a dark line crossing the pale cheek'},
 'Phainopepla': {'Male': 'glossy jet black with a shaggy crest and red eye',
                 'Female/juvenile': 'plain gray with a shaggy crest and red eye'},
 "Brewer's Blackbird": {'Male': 'glossy black with purple head sheen and pale yellow eye',
                        'Female/Juvenile': 'plain gray-brown all over with a dark eye'},
 'Summer Tanager': {'Adult Male': 'entirely rosy red with a pale stout bill',
                    'Female': 'uniform mustard yellow-olive with a pale stout bill',
                    'Immature Male': 'patchy mix of yellow-olive and red blotches'},
 'Purple Martin': {'Adult male': 'entirely glossy blue-purple, the darkest of swallows',
                   'Female/juvenile': 'dusky gray below with a pale collar, dark above'},
 'Common Yellowthroat': {'Adult Male': 'broad black bandit mask above a bright yellow throat',
                         'Female/immature male': 'plain olive with a yellow throat and no black mask'},
 'American Redstart': {'Adult Male': 'glossy black with flaming orange patches on wings and tail',
                       'Female/juvenile': 'gray-olive with lemon yellow wing and tail patches'},
 'Magnolia Warbler': {'Breeding male': 'black mask and heavy black necklace streaks on yellow',
                      'Female/immature male': 'gray head, faint streaks, yellow below, white tail band'},
 'Bay-breasted Warbler': {'Breeding male': 'chestnut crown, throat and flanks, black face, buff neck',
                          'Female, Nonbreeding male, Immature': 'greenish above, pale below with buffy flanks, faint streaks'},
 'Chestnut-sided Warbler': {'Breeding male': 'yellow cap with a broad chestnut stripe down the flank',
                            'Female/immature male': 'lime green cap, white eye ring, plain white underparts'},
 'Blackpoll Warbler': {'Breeding male': 'solid black cap, white cheek, black-streaked white sides',
                       'Female/juvenile': 'olive with faint streaks, orange legs, no black cap'},
 'Black-throated Blue Warbler': {'Adult Male': 'deep blue above, black face and throat, white belly',
                                 'Female/Immature male': 'plain olive-brown with a small white wing spot'},
 'Lark Bunting': {'Breeding male': 'entirely coal black with a large white wing patch',
                  'Female/Nonbreeding male': 'brown and heavily streaked with a buffy wing patch'},
 'Scarlet Tanager': {'Breeding Male': 'brilliant scarlet body with jet black wings and tail',
                     'Female/Nonbreeding Male': 'olive-yellow body with darker olive or black wings'},
 'Western Tanager': {'Breeding Male': 'flaming red head, yellow body, black back and wings',
                     'Female/Nonbreeding Male': 'yellow-olive with gray back and two pale wing bars'},
 'Northern Cardinal': {'Adult Male': 'entirely brilliant red with a black face and red crest',
                       'Female/Juvenile': 'buff-brown with red on the crest, wings and tail'},
 'Rose-breasted Grosbeak': {'Adult Male': 'black and white with a triangular rose-red breast patch',
                            'Female/immature male': 'brown and heavily streaked with a bold white eyebrow'},
 'Black-headed Grosbeak': {'Adult Male': 'black head, burnt orange breast and collar, black wings',
                           'Female/immature male': 'brown with a striped head and buffy orange breast'},
 'Blue Grosbeak': {'Adult Male': 'deep blue with two broad chestnut wing bars',
                   'Female/juvenile': 'warm cinnamon brown with buffy wing bars, huge bill'},
 'Lazuli Bunting': {'Adult Male': 'turquoise blue head, orange breast band, white belly',
                    'Female/juvenile': 'plain warm brown with a hint of blue and faint wing bars'},
 'Indigo Bunting': {'Adult Male': 'entirely vivid indigo blue with no other markings',
                    'Female/juvenile': 'plain warm brown with faint blurry breast streaking'},
 'Painted Bunting': {'Adult Male': 'blue head, red underparts, lime green back',
                     'Female/juvenile': 'uniform bright lime green with a pale eye ring'},
 'Bobolink': {'Breeding male': 'black below with a creamy nape and white back patches',
              'Female/juvenile/nonbreeding male': 'warm buff and streaked with bold dark crown stripes'},
 'Red-winged Blackbird': {'Male': 'glossy black with red and yellow shoulder epaulets',
                          'Female/juvenile': 'dark brown and heavily streaked with a pale eyebrow'},
 'Yellow-headed Blackbird': {'Adult Male': 'black body with a blazing yellow head and breast',
                             'Female/Immature Male': 'dusky brown with a dull yellow throat and face'},
 'Brown-headed Cowbird': {'Male': 'glossy black body with a contrasting brown head',
                          'Female/Juvenile': 'plain dull gray-brown all over with faint streaking'},
 'Hooded Oriole': {'Adult male': 'orange-yellow with a black bib and long curved bill',
                   'Female/Immature male': 'olive-yellow below, grayish back, long curved bill'},
 "Bullock's Oriole": {'Adult male': 'bright orange face with a black eye line and crown',
                      'Female/Immature male': 'yellow head and breast, gray belly, white wing bars'},
 'Baltimore Oriole': {'Adult male': 'brilliant flame orange with a solid black hood',
                      'Female/Immature male': 'yellow-orange below, olive-brown back, no black hood'},
 'Pine Grosbeak': {'Adult Male': 'plump rosy pink-red bird with gray wings and white bars',
                   'Female/juvenile': 'gray body with a mustard yellow or russet head'},
 'Purple Finch': {'Adult Male': 'raspberry red wash over the head, breast and back',
                  'Female/immature': 'brown and streaked with a bold white eyebrow and cheek stripe'},
 "Cassin's Finch": {'Adult Male': 'bright red peaked crown fading to pink breast, streaked back',
                    'Female/immature': 'crisply streaked brown and white with a peaked head'},
 'House Finch': {'Adult Male': 'red forehead, throat and rump, brown streaked flanks',
                 'Female/immature': 'plain gray-brown with blurry streaks and no face pattern'},
 'Red Crossbill': {'Adult Male': 'brick red body with dark wings and crossed bill tips',
                   'Female/juvenile': 'olive-yellow body with dark wings and crossed bill tips'},
 'White-winged Crossbill': {'Adult Male': 'pink-red body with black wings and two white wing bars',
                            'Female/juvenile': 'streaky olive-yellow with black wings and white wing bars'},
 'Lesser Goldfinch': {'Adult Male': 'glossy black cap and back over bright yellow underparts',
                      'Female/juvenile': 'dull olive above, pale yellow below, white wing marks'},
 'American Goldfinch': {'Breeding Male': 'brilliant canary yellow with a black forehead and wings',
                        'Female/Nonbreeding Male': 'drab olive-buff with blackish wings and pale wing bars'},
 'Evening Grosbeak': {'Adult Male': 'yellow body, dark head with a yellow brow, huge pale bill',
                      'Female/Juvenile': 'gray body with yellow-green tinges and white wing patches'},
 'House Sparrow': {'Male': 'gray crown, black bib, chestnut nape, white cheek',
                   'Female/Juvenile': 'plain dull brown with a buffy eyebrow and streaked back'},
 'Vermilion Flycatcher': {'Adult male': 'blazing vermilion crown and underparts with a dark mask',
                          'Female, immature': 'gray-brown above, whitish streaked breast, pinkish belly'},
 'Orchard Oriole': {'Adult Male': 'deep brick chestnut body with a solid black hood',
                    'Immature Male': 'greenish yellow with a small neat black throat patch',
                    'Female/Juvenile': 'uniform greenish yellow with two white wing bars'},
}


def build():
    out = {}
    for sp, variants in D.items():
        for var, text in variants.items():
            out[f'{sp} ({var})'] = f', {text}'
    return out


def shuffled(desc, seed=0):
    """Phep DOI CHUNG cho ket luan "mo ta rieng loai > mo ta chung".

    Hoan vi mo ta GIUA CAC LOAI nhung GIU NGUYEN vai tro male/female. Baltimore
    Oriole (male) nhan mo ta cua mot loai khac, cung la con trong.

    Doc ket qua:
      - neu ban hoan vi cho ket qua NGANG ban dung -> cai lai khong den tu NOI DUNG
        ma chi tu viec moi la co mot chuoi text rieng. Ket luan phai rut lai.
      - neu tut ve muc mo ta CHUNG (T0d/T3a) -> noi dung that su mang thong tin.
    """
    import random
    import re
    role = {}
    for leaf in desc:
        var = re.search(r'\((.*)\)\s*$', leaf).group(1).lower()
        role[leaf] = 'female' if 'female' in var else 'male'
    out = {}
    rng = random.Random(seed)
    for r in ('male', 'female'):
        keys = sorted(k for k in desc if role[k] == r)
        vals = [desc[k] for k in keys]
        while True:                              # khong de la nao giu mo ta cu
            perm = vals[:]
            rng.shuffle(perm)
            if all(a != b for a, b in zip(vals, perm)) or len(keys) < 2:
                break
        out.update(dict(zip(keys, perm)))
    return out


def main():
    check = '--check' in sys.argv
    desc = build()
    import pandas as pd
    leaf = pd.read_csv(os.path.join(zs_env.DATA_DIR, 'leaf_variants.csv'),
                       keep_default_na=False)
    known = set(leaf.leaf_name)
    bad = [k for k in desc if k not in known]

    p = os.path.join(zs_env.DATA_DIR, 'leaf_descriptors.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(desc, f, indent=1, ensure_ascii=False)
    ps = os.path.join(zs_env.DATA_DIR, 'leaf_descriptors_shuffled.json')
    with open(ps, 'w', encoding='utf-8') as f:
        json.dump(shuffled(desc), f, indent=1, ensure_ascii=False)

    import splits
    import prompts
    df = prompts.load_tables()
    un = splits.variant_split(df)
    sub = df.sort_values('class_id', key=lambda s: s.astype(int)).reset_index(drop=True)
    unseen_names = set(sub.loc[un, 'leaf_name'])

    print(f'  mo ta viet ra          : {len(desc)}')
    print(f'  ten la KHONG khop       : {len(bad)}  (can 0)')
    for b in bad:
        print(f'      !! {b}')
    print(f'  phu 81 la unseen        : {len(unseen_names & set(desc))}/{len(unseen_names)}')
    print(f'  phu 288 la bien the     : {len(set(desc) & set(leaf[leaf.variant_raw != ""].leaf_name))}/288')
    if check:
        for k in sorted(desc):
            print(f'  {k:52s}{desc[k]}')
    print(f'\n-> {p}')


if __name__ == '__main__':
    main()
