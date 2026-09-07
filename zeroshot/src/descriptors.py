"""T3a — dịch các trục biến thể đã parse thành NGÔN NGỮ THỊ GIÁC.

Động cơ (đo được, không phải suy đoán): ở T2, thêm cụm biến thể thô vào prompt chỉ
nâng độ chính xác biến thể từ 56.21% (mốc "luôn đoán biến thể phổ biến nhất, không
nhìn ảnh") lên 58.18% — tức +1.97 điểm. Text encoder nhận được một CHUỖI khác nhau
cho mỗi lá, nhưng chuỗi đó ("adult male") không nói gì về việc con chim TRÔNG thế nào.

T3a chỉ dùng kiến thức ĐẠI CƯƠNG về lưỡng hình giới tính ở chim — trống bộ lông sinh
sản sặc sỡ tương phản, mái/non xỉn nâu nhiều vệt — nên **không có rủi ro bịa đặt về
loài cụ thể**. Biến độc lập sạch: T3a ăn thua thì vấn đề của T2 là thiếu từ vựng thị
giác; T3a không ăn thua thì vấn đề nằm ở khả năng NỐI text với ảnh.

RÀNG BUỘC ĐỘ DÀI: tokenizer CLIP chỉ có 77 token. Tiền tố T2 đã chiếm tới 52.
Vì vậy mỗi cụm dưới đây <= 12 từ và ghép tối đa 2 cụm.
"""

MALE_BREEDING = 'a bright breeding male with bold saturated colors'
MALE_PLAIN = 'an adult male, brighter and more boldly marked than the female'
FEMALE = 'a dull female or immature, brownish and streaked, with weak markings'
IMMATURE = 'a young bird, duller and browner than the adult, often streaked'
NONBREEDING = 'in drab winter plumage of dull brown and gray tones'
ECLIPSE = 'in dull eclipse plumage resembling the brown female'
ADULT = 'a clean full adult'

MORPH = {
    'dark': 'a dark morph, almost uniformly sooty blackish',
    'light': 'a light morph, whitish below against darker upperparts',
    'white': 'a white morph, almost entirely white with black wingtips',
    'blue': 'a blue morph, dark blue-gray body with a white head',
}

# Nhóm phụ loài: chỉ nêu đặc điểm NHÌN THẤY ĐƯỢC, không suy diễn thêm.
FORM = {
    'myrtle': 'the Myrtle form, white throat and dark cheek patch',
    "audubon's": 'the Audubon form, with a yellow throat',
    'slate-colored': 'the slate-colored form, plain dark gray hood and back',
    'oregon': 'the Oregon form, dark hood, brown back, pinkish sides',
    'pink-sided': 'the pink-sided form, gray hood and broad pink flanks',
    'white-winged': 'the white-winged form, pale gray with white wing bars',
    'red-backed': 'the red-backed form, with a rufous back patch',
    'gray-headed': 'the gray-headed form, pale gray head and rufous back',
    'red-shafted': 'the red-shafted form, salmon-red wing and tail shafts',
    'yellow-shafted': 'the yellow-shafted form, bright yellow wing and tail shafts',
    'thick-billed': 'the thick-billed form, with a heavy stubby bill',
    'tan-striped': 'the tan-striped form, tan and brown head stripes',
    'white-striped': 'the white-striped form, crisp black and white head stripes',
    'sooty': 'the sooty form, uniformly dark sooty-brown above',
    'red': 'the red form, rufous above with heavy rufous breast spots',
}

MAX_PARTS = 2                                    # giữ prompt trong 77 token


def parts_from_axes(sex, age, season, morph, form):
    """Các trục đã parse -> danh sách cụm mô tả, quan trọng nhất đứng trước."""
    out = []
    if morph:
        out += [MORPH[m] for m in morph if m in MORPH]
    if form:
        out += [FORM[f] for f in form if f in FORM]

    has_f, has_m = 'female' in sex, 'male' in sex
    young = bool({'immature', 'juvenile'} & set(age))
    if has_f:
        out.append(FEMALE)                       # lớp gộp mái + trống-giống-mái
    elif has_m:
        if 'breeding' in season:
            out.append(MALE_BREEDING)
        elif 'eclipse' in season:
            out.append(ECLIPSE)
        elif young:
            out.append(IMMATURE)
        else:
            out.append(MALE_PLAIN)
    elif young:
        out.append(IMMATURE)

    if not has_f:                                # với lớp female-side, FEMALE đã bao hàm
        if 'nonbreeding' in season and 'breeding' not in season:
            out.append(NONBREEDING)
        elif 'eclipse' in season and not has_m:
            out.append(ECLIPSE)
    if 'adult' in age and not out:
        out.append(ADULT)
    return out[:MAX_PARTS]


def generic_phrase(sex, age, season, morph, form):
    """-> một cụm bổ nghĩa, hoặc '' nếu lá không có biến thể."""
    parts = parts_from_axes(sex, age, season, morph, form)
    return ', ' + '; '.join(parts) if parts else ''
