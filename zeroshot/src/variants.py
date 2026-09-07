"""Parser cho phần trong ngoặc đơn của tên lớp lá NABirds.

Vì sao cần: 555 lá gộp về 404 species, nên prompt chỉ mang tên loài bị chặn trần
**78.75%** top-1 ở bài 555 lớp (xem README mục 1.2). Muốn vượt trần đó thì text
phải mang được thông tin giới tính / tuổi / mùa / morph / phụ loài — thông tin đó
nằm trong ngoặc đơn của tên lá:

    Yellow-rumped Warbler (Breeding Myrtle)
    Common Eider (Female/juvenile)
    Dark-eyed Junco (Oregon)

Có **60 chuỗi ngoặc đơn khác nhau** (52 nếu bỏ qua hoa/thường) trên 288/555 lá.
Dữ liệu bẩn: `Adult Male` vs `Adult male`, `Female/Immature male` vs
`Female/Immature Male`, `Adult ` thừa dấu cách.

Chuỗi thường là **phép tuyển**: `Female/Eclipse male` = "con mái HOẶC con trống bộ
lông eclipse". Parser giữ nguyên nghĩa tuyển đó (mỗi trường là một tuple) thay vì
ép về một giá trị duy nhất.
"""
import re

# Morph màu — luôn viết dạng "<màu> morph"
MORPH_RE = re.compile(r'\b(dark|light|white|blue)\s+morph\b')

# Nhóm phụ loài / dạng bộ lông có tên riêng. Khớp chuỗi dài trước ('red-backed'
# phải thắng 'red') nên sắp theo độ dài giảm dần lúc dùng.
FORMS = (
    "audubon's", 'red-backed', 'gray-headed', 'red-shafted', 'yellow-shafted',
    'thick-billed', 'slate-colored', 'white-winged', 'white-striped',
    'tan-striped', 'pink-sided', 'myrtle', 'oregon', 'sooty', 'red',
)
FORMS_BY_LEN = tuple(sorted(FORMS, key=len, reverse=True))

# Giữ hoa cho danh từ riêng khi sinh câu
PROPER = {'myrtle': 'Myrtle', "audubon's": "Audubon's", 'oregon': 'Oregon'}

SEASONS = (('breeding', 'breeding'), ('nonbreeding', 'nonbreeding'),
           ('winter', 'nonbreeding'), ('summer', 'breeding'),
           ('eclipse', 'eclipse'))
AGES = ('subadult', 'immature', 'juvenile', 'adult')


def parse_variant(raw):
    """Chuỗi trong ngoặc đơn -> dict các trục ngữ nghĩa + câu dùng cho prompt.

    Trả về `sex`/`age`/`season`/`morph`/`form` là **tuple** (rỗng = không có),
    vì một chuỗi có thể mang nhiều giá trị cùng lúc (`Adult, Subadult`).
    `known=False` nghĩa là không nhận ra trục nào -> cần xem lại bằng tay.
    """
    s = ' '.join(raw.strip().lower().split())

    morph = tuple(MORPH_RE.findall(s))
    rest = MORPH_RE.sub(' ', s)

    form = []
    for f in FORMS_BY_LEN:
        if re.search(rf'(?<![\w-]){re.escape(f)}(?![\w-])', rest):
            form.append(f)
            rest = re.sub(rf'(?<![\w-]){re.escape(f)}(?![\w-])', ' ', rest)
    form = tuple(form)

    season = tuple(dict.fromkeys(
        norm for kw, norm in SEASONS if re.search(rf'\b{kw}\b', rest)))
    age = tuple(a for a in AGES if re.search(rf'\b{a}\b', rest))

    sex = []
    if re.search(r'\bfemale\b', rest):
        sex.append('female')
    if re.search(r'\bmale\b', rest):          # \b không khớp 'male' trong 'female'
        sex.append('male')
    sex = tuple(sex)

    return {
        'raw': raw, 'sex': sex, 'age': age, 'season': season,
        'morph': morph, 'form': form,
        'known': bool(sex or age or season or morph or form),
        'phrase': _phrase(s, form),
    }


def _phrase(s, form):
    """Chuỗi đã lower -> cụm tiếng Anh tự nhiên cho prompt.

    `/` và `,` là phép tuyển -> ' or '. Thêm ' form' chỉ khi tên phụ loài đứng
    CUỐI chuỗi, nếu không câu sẽ sai chỗ ('tan-striped or immature form').
    """
    p = re.sub(r'\s*[/,]\s*', ' or ', s)
    for k, v in PROPER.items():
        p = re.sub(rf'(?<![\w-]){re.escape(k)}(?![\w-])', v, p)
    if form and any(s.endswith(f) for f in form):
        p += ' form'
    return p


def split_leaf_name(leaf_name):
    """'Common Eider (Female/juvenile)' -> ('Common Eider', 'Female/juvenile').

    Lá không có ngoặc đơn -> (tên, None).
    """
    m = re.match(r'^(.*?)\s*\((.*)\)\s*$', leaf_name)
    return (m.group(1).strip(), m.group(2).strip()) if m else (leaf_name.strip(), None)


def is_female_side(rec):
    """Lá có chứa cá thể mái / non không? Dùng để dựng split unseen 'biến thể'.

    True cho cả `Female` lẫn `Female/Eclipse male` (lớp gộp mái + trống eclipse),
    vì cả hai đều là bộ lông mà một model chỉ học từ lá 'male' chưa từng thấy.
    """
    return rec is not None and 'female' in rec['sex']


def is_male_only(rec):
    """Lá chỉ gồm cá thể trống (`Adult male`, `Breeding male`, `Winter male`)."""
    return rec is not None and rec['sex'] == ('male',)
