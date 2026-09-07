"""Model factory cho NABirds 555 lớp.

Tất cả trọng số pretrain tải về `<repo>/models` (đặt TORCH_HOME / HF_HOME).

Mỗi model trả về (module, default_img_size). Chỉ lớp phân loại cuối được thay
(2048/768 -> 555); phần backbone giữ nguyên cấu hình chuẩn vì trọng số pretrain
chỉ tồn tại cho các cấu hình đó.
"""
import os

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ.setdefault('TORCH_HOME', MODEL_DIR)
os.environ.setdefault('HF_HOME', os.path.join(MODEL_DIR, 'hf'))
# Máy này có TensorFlow build theo NumPy 1.x nhưng NumPy hiện tại là 2.x. `apply_lora`
# import peft -> transformers -> `import tensorflow` -> `numpy.core.umath failed to
# import` và giết cả tiến trình. Tắt nhánh TF là đủ, KHÔNG gỡ TensorFlow.
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

# Log của các script này có tiếng Việt. Khi stdout bị redirect ra file, Windows mặc
# định cp1252 -> UnicodeEncodeError giết cả run đang train.
import sys  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

from torchvision import models as tvm  # noqa: E402  (phải sau khi set TORCH_HOME)

NUM_CLASSES = 555


# ---------------------------------------------------------------------------
# 1. CNN tự xây (from scratch) — baseline để thấy khoảng cách với pretrain
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """~5M params. VGG-style, BN + GAP head.

    Input  (B,3,224,224)
    Stem   Conv3x3 3->32 s2                 -> 112
    Blk1-4 [Conv3x3 x2 + MaxPool2]          -> 56 -> 28 -> 14 -> 7
    Head   GAP -> Dropout(0.3) -> Linear(512,555)
    """

    def __init__(self, num_classes=NUM_CLASSES, widths=(32, 64, 128, 256, 512)):
        super().__init__()
        w0 = widths[0]
        layers = [nn.Conv2d(3, w0, 3, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(w0), nn.ReLU(inplace=True)]
        c_in = w0
        for c_out in widths[1:]:
            layers += [
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.drop(x))


# ---------------------------------------------------------------------------
# 2. Registry
# ---------------------------------------------------------------------------
def _resnet(depth, pretrained):
    fn = {50: tvm.resnet50, 101: tvm.resnet101}[depth]
    w = {50: tvm.ResNet50_Weights.IMAGENET1K_V2,
         101: tvm.ResNet101_Weights.IMAGENET1K_V2}[depth] if pretrained else None
    m = fn(weights=w)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m, 224


def _vit_b16(pretrained):
    w = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.vit_b_16(weights=w)
    m.heads.head = nn.Linear(m.heads.head.in_features, NUM_CLASSES)
    return m, 224


def _vit_b16_in21k(pretrained, img_size=None):
    """ViT-B/16 pretrain ImageNet-21k (timm) — đây là checkpoint mà TransFG và
    các paper FGVC dùng, mạnh hơn hẳn bản ImageNet-1k của torchvision.

    `img_size` phải truyền vào LÚC DỰNG model: ViT có positional embedding gắn
    chặt với số patch. Đổi độ phân giải chỉ ở transform ảnh mà không báo cho timm
    thì hoặc lỗi shape, hoặc (tệ hơn) chạy sai lặng lẽ. timm nội suy pos-embed
    khi nhận `img_size` khác cfg gốc."""
    import timm
    kw = {'img_size': img_size} if img_size else {}
    m = timm.create_model('vit_base_patch16_224.augreg_in21k',
                          pretrained=pretrained, num_classes=NUM_CLASSES, **kw)
    return m, img_size or 224


# CNN hiện đại qua timm: (model_id, img_size mặc định, tên lớp phân loại).
# `head` lấy từ `default_cfg['classifier']` của timm — ghi tường minh ở đây để
# `head_parameter_names` không phải dựng model mới chỉ để tra tên.
#
# Vì sao chọn đúng ba cái này:
#   convnext_tiny  fb_in1k  — cùng nguồn pretrain + cùng 224 với resnet50 đã chạy,
#                             nên chênh lệch quy được về THIẾT KẾ KIẾN TRÚC.
#   convnext_tiny  fb_in22k — cùng kiến trúc với dòng trên, chỉ đổi nguồn pretrain,
#                             lặp lại đúng trục IN1k -> IN21k đã đo trên ViT (+6.67).
#   tf_efficientnetv2_s in21k — họ CNN khác, cũng IN21k: kiểm chứng kết luận của
#                             trục pretrain không phải đặc thù ConvNeXt.
# Lưu ý: cfg gốc của tf_efficientnetv2_s là 300px; ở đây ép 224 để nằm chung trục
# độ phân giải với mọi run khác, nên con số của nó là cận DƯỚI.
TIMM_MODELS = {
    'convnext_tiny':       ('convnext_tiny.fb_in1k', 224, 'head.fc'),
    'convnext_tiny_in22k': ('convnext_tiny.fb_in22k', 224, 'head.fc'),
    'efficientnetv2_s':    ('tf_efficientnetv2_s.in21k', 224, 'classifier'),
    # Swin-T (27.9M) là đối thủ trực tiếp cùng cỡ của ConvNeXt-T (28.2M) trong
    # chính paper ConvNeXt. Lấy bản IN22k để so ngang với `convnext_tiny_in22k`:
    # cùng cỡ, cùng nguồn pretrain, cùng 224 -> chênh lệch quy về CNN vs transformer.
    'swin_tiny_in22k':     ('swin_tiny_patch4_window7_224.ms_in22k', 224, 'head.fc'),
    # ViT-S (21.9M) CÙNG checkpoint family `augreg_in21k` với vit_b_16_in21k
    # (86.2M) -> trục QUY MÔ trong họ ViT, không lẫn nguồn pretrain lẫn recipe.
    'vit_small_in21k':     ('vit_small_patch16_224.augreg_in21k', 224, 'head'),
}


def _timm_model(name, pretrained, img_size=None):
    """Model timm chỉ thay lớp phân loại cuối (kể cả checkpoint 21k-way)."""
    import timm
    model_id, default_size, _ = TIMM_MODELS[name]
    kw = {'img_size': img_size} if img_size and img_size != default_size else {}
    m = timm.create_model(model_id, pretrained=pretrained,
                          num_classes=NUM_CLASSES, **kw)
    return m, img_size or default_size


# Model DL "cơ bản" khác qua torchvision: (builder, weights enum, img_size, head).
# Mục đích: trải dài trục THỜI GIAN kiến trúc (2014 VGG -> 2022 ConvNeXt) và trục
# QUY MÔ tham số (5.5M MobileNetV3 -> 134M VGG16), tất cả cùng pretrain IN1k @224
# nên so được trực tiếp với resnet50_224 (78.61).
TV_MODELS = {
    'vgg16_bn':           ('vgg16_bn', 'VGG16_BN_Weights', 224, 'classifier.6'),
    'densenet121':        ('densenet121', 'DenseNet121_Weights', 224, 'classifier'),
    'mobilenet_v3_large': ('mobilenet_v3_large', 'MobileNet_V3_Large_Weights', 224,
                           'classifier.3'),
    'resnext50_32x4d':    ('resnext50_32x4d', 'ResNeXt50_32X4D_Weights', 224, 'fc'),
    # Hai mốc đầu của trục thời gian. AlexNet 2012 là điểm khởi đầu của cả làn sóng
    # deep learning thị giác; nó KHÔNG có BatchNorm nên phải hạ lr (xem DEFAULT_LR).
    'alexnet':            ('alexnet', 'AlexNet_Weights', 224, 'classifier.6'),
    'efficientnet_b0':    ('efficientnet_b0', 'EfficientNet_B0_Weights', 224,
                           'classifier.1'),
}


def _tv_model(name, pretrained):
    fn_name, w_name, img_size, head = TV_MODELS[name]
    w = getattr(tvm, w_name).IMAGENET1K_V1 if pretrained else None
    m = getattr(tvm, fn_name)(weights=w)
    mod, attr = m, head
    if '.' in head:                       # vd 'classifier.6' -> nn.Sequential
        parent, attr = head.rsplit('.', 1)
        for part in parent.split('.'):
            mod = getattr(mod, part)
    old = getattr(mod, attr)
    setattr(mod, attr, nn.Linear(old.in_features, NUM_CLASSES))
    return m, img_size


# ---------------------------------------------------------------------------
# BioCLIP: image tower của foundation model sinh học, dùng cho bài CÓ GIÁM SÁT
# ---------------------------------------------------------------------------
# Đây là điểm thứ TƯ trên trục nguồn pretrain: không có -> IN1k -> IN21k -> ToL-10M.
# Kiến trúc y hệt `vit_b_16_in21k` (ViT-B/16, 86M) nên phải chạy CÙNG recipe
# (SGD, lr 0.001/0.01, 25 epoch, bs 64) thì phép so mới quy được về nguồn pretrain.
#
# HAI khác biệt bắt buộc so với các model khác:
#   1. Chuẩn hoá ảnh là của CLIP, KHÔNG phải ImageNet -> xem `norm_stats()`.
#   2. `model.visual` trả về embedding 512-d đã chiếu; head là Linear(512, 555).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class OpenClipClassifier(nn.Module):
    """Image tower của một model open_clip + lớp phân loại tuyến tính."""

    def __init__(self, visual, dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.visual = visual
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        feat = self.visual(x)
        if isinstance(feat, (tuple, list)):        # output_tokens=True
            feat = feat[0]
        return self.fc(feat)


def _bioclip(pretrained):
    """Cờ `pretrained` bị BỎ QUA — và đó là chủ ý.

    open_clip nạp `hf-hub:` là nạp cả config lẫn trọng số, không tách ra được.
    Quan trọng hơn: "BioCLIP không pretrain" chẳng là gì ngoài một ViT-B/16 khởi
    tạo ngẫu nhiên, mà trục đó đã có `vit_b_16 --no-pretrained` rồi. Nếu raise ở
    nhánh này thì `src/ensemble.py` chết, vì nó dựng lại khung bằng
    `pretrained=False` trước khi nạp checkpoint đã train.
    """
    del pretrained
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        'hf-hub:imageomics/bioclip',
        cache_dir=os.path.join(MODEL_DIR, 'openclip'))
    dim = model.visual.output_dim
    return OpenClipClassifier(model.visual, dim), 224


def _googlenet(pretrained):
    """Inception-v1 (2014). Không dùng chung `_tv_model` được vì hai nhánh phụ.

    Bẫy: KHÔNG được truyền `aux_logits=False` cùng với `weights` — torchvision ép
    `aux_logits=True` để nạp checkpoint rồi so với giá trị người gọi đưa vào, và
    raise `ValueError: expected value True but got False`. Cách đúng là **không
    truyền gì cả**: mặc định `aux_logits=False` khiến nó nạp xong rồi tự gỡ
    `aux1`/`aux2`. Kết quả: thân đã pretrain, không nhánh phụ, forward trả về MỘT
    tensor ở cả train lẫn eval — không cần nhánh xử lý riêng trong vòng lặp train
    như `inception_v3`.
    """
    # `transform_input` phải khớp giữa hai nhánh — GoogLeNet dính ĐÚNG bug của
    # `_inception_v3`: torchvision ép True khi có weights, mặc định False khi không.
    # Đo được: dựng bằng pretrained=False rồi nạp checkpoint cho 65.10% thay vì
    # 72.21%, mất 7.11 điểm không một cảnh báo.
    if pretrained:
        m = tvm.googlenet(weights=tvm.GoogLeNet_Weights.IMAGENET1K_V1,
                          transform_input=True)
    else:
        m = tvm.googlenet(weights=None, aux_logits=False, transform_input=True,
                          init_weights=True)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m, 224


def _inception_v3(pretrained):
    """Inception-v3 dùng input 299x299 và có nhánh phụ AuxLogits.

    torchvision ép transform_input=True khi load trọng số pretrain (trọng số
    port từ TensorFlow), nên vẫn dùng chuẩn hoá ImageNet bình thường ở dataset.

    BẮT BUỘC truyền `transform_input=True` tường minh. Nếu không:
      pretrained=True  -> torchvision ép transform_input=True
      pretrained=False -> mặc định transform_input=False
    `transform_input` là một thuộc tính bool, KHÔNG nằm trong state_dict, nên
    `load_state_dict` vẫn thành công và model chạy SAI LẶNG LẼ. Đo được: dựng
    khung bằng pretrained=False rồi nạp checkpoint đã train cho **74.87%** thay
    vì **80.93%** — mất 6.06 điểm mà không có lấy một cảnh báo. Đây đúng là
    đường mà `src/ensemble.py` đi.
    """
    w = tvm.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.inception_v3(weights=w, aux_logits=True, transform_input=True,
                         init_weights=not pretrained)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, NUM_CLASSES)
    return m, 299


# Mọi builder nhận (pretrained, img_size). Model toàn tích chập bỏ qua img_size vì
# kích thước input không đổi kiến trúc; ViT/Swin thì BẮT BUỘC nhận (xem _vit_b16_in21k).
BUILDERS = {
    'cnn_scratch':   lambda pretrained, img_size=None: (SimpleCNN(), img_size or 224),
    'resnet50':      lambda pretrained, img_size=None: _resnet(50, pretrained),
    'resnet101':     lambda pretrained, img_size=None: _resnet(101, pretrained),
    'vit_b_16':      lambda pretrained, img_size=None: _vit_b16(pretrained),
    'vit_b_16_in21k': _vit_b16_in21k,
    'inception_v3':  lambda pretrained, img_size=None: _inception_v3(pretrained),
    'bioclip':       lambda pretrained, img_size=None: _bioclip(pretrained),
    'googlenet':     lambda pretrained, img_size=None: _googlenet(pretrained),
    **{k: (lambda p, img_size=None, _k=k: _timm_model(_k, p, img_size))
       for k in TIMM_MODELS},
    **{k: (lambda p, img_size=None, _k=k: _tv_model(_k, p)) for k in TV_MODELS},
}

MODEL_NAMES = list(BUILDERS)


def _init_head(module):
    """Khởi tạo lại lớp phân loại mới với trọng số nhỏ.

    Với 555 lớp, loss lúc khởi tạo phải xấp xỉ ln(555) = 6.32. Đo pre-flight cho
    thấy head mặc định của timm cho `vit_b_16_in21k` bắt đầu ở 8.5 (logit quá
    lớn), trong khi các model khác đều ~6.4. Gradient lớn ở vài step đầu dễ phá
    trọng số pretrain của backbone. Khởi tạo nhỏ + bias 0 để mọi model cùng xuất
    phát từ phân phối gần đều.
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_model(name, pretrained=True, img_size=None):
    if name not in BUILDERS:
        raise ValueError(f'model phải thuộc {MODEL_NAMES}, nhận được {name!r}')
    if name == 'cnn_scratch':
        pretrained = False
    model, img_size = BUILDERS[name](pretrained, img_size)
    for h in head_parameter_names(model, name):
        mod = model
        for part in h.rstrip('.').split('.'):
            mod = getattr(mod, part, None)
            if mod is None:
                break
        if mod is not None:
            _init_head(mod)
    model.default_img_size = img_size
    model.model_name = name
    model.is_inception = (name == 'inception_v3')
    model.norm_mean, model.norm_std = norm_stats(name)
    return model, img_size


def norm_stats(name):
    """(mean, std) để chuẩn hoá ảnh. BioCLIP được pretrain bằng chuẩn hoá CLIP;
    đưa ảnh chuẩn hoá kiểu ImageNet vào là lệch phân phối đầu vào ngay từ epoch 0.
    Mọi model còn lại đều pretrain trên ImageNet nên dùng thống kê ImageNet."""
    if name == 'bioclip':
        return CLIP_MEAN, CLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD


def head_parameter_names(model, name):
    """Tên các tham số thuộc lớp phân loại cuối (dùng lr lớn hơn backbone)."""
    if name in ('resnet50', 'resnet101'):
        return ['fc.']
    if name == 'inception_v3':
        return ['fc.', 'AuxLogits.fc.']
    if name == 'vit_b_16':
        return ['heads.head.']
    if name == 'vit_b_16_in21k':
        return ['head.']
    if name in TIMM_MODELS:
        return [TIMM_MODELS[name][2] + '.']
    if name in TV_MODELS:
        return [TV_MODELS[name][3] + '.']
    if name == 'googlenet':
        return ['fc.']
    return ['fc.']                     # resnet50/101, bioclip (OpenClipClassifier)


def param_groups(model, name, lr_backbone, lr_head, weight_decay):
    """2 nhóm lr (head cao hơn) + bỏ weight decay cho bias/norm."""
    head_keys = head_parameter_names(model, name)
    groups = {k: {'decay': [], 'no_decay': []} for k in ('head', 'backbone')}
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        which = 'head' if any(pname.startswith(h) or f'.{h}' in pname
                              for h in head_keys) else 'backbone'
        bucket = 'no_decay' if (p.ndim <= 1 or pname.endswith('.bias')) else 'decay'
        groups[which][bucket].append(p)
    out = []
    for which, lr in (('backbone', lr_backbone), ('head', lr_head)):
        if groups[which]['decay']:
            out.append({'params': groups[which]['decay'], 'lr': lr,
                        'weight_decay': weight_decay, 'name': f'{which}_decay'})
        if groups[which]['no_decay']:
            out.append({'params': groups[which]['no_decay'], 'lr': lr,
                        'weight_decay': 0.0, 'name': f'{which}_nodecay'})
    return out


def set_num_classes(model, name, num_classes):
    """Thay lớp phân loại cuối bằng lớp khác số đầu ra (dùng cho transfer sang CUB).

    Dùng chính `head_parameter_names` để tìm module head nên không phải viết lại
    bảng tên head lần thứ hai cho mỗi kiến trúc.
    """
    for h in head_parameter_names(model, name):
        parts = h.rstrip('.').split('.')
        mod = model
        for part in parts[:-1]:
            mod = getattr(mod, part)
        old_layer = getattr(mod, parts[-1])
        new_layer = nn.Linear(old_layer.in_features, num_classes)
        _init_head(new_layer)
        setattr(mod, parts[-1], new_layer)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# LoRA chỉ gắn được vào nn.Linear. timm ViT gộp q/k/v thành MỘT `nn.Linear` tên
# `qkv` nên gắn được; torchvision ViT và open_clip dùng nn.MultiheadAttention với
# q/k/v nằm trong `in_proj_weight` (nn.Parameter thô) nên KHÔNG gắn được.
# `attn.proj` chứ không phải `proj`: peft khớp theo hậu tố tên module, nên `proj`
# trúng cả `patch_embed.proj` (Conv2d nhúng patch) — không phải chỗ ta muốn adapt.
LORA_TARGETS = {'vit_b_16_in21k': ['qkv', 'attn.proj', 'fc1', 'fc2']}


def apply_lora(model, name, rank):
    """Đóng băng backbone, gắn adapter LoRA, giữ lớp phân loại cuối train được."""
    if name not in LORA_TARGETS:
        raise ValueError(f'--lora-rank chua ho tro {name}; chi co '
                         f'{list(LORA_TARGETS)} (xem chu thich LORA_TARGETS)')
    from peft import LoraConfig, get_peft_model
    head = [h.rstrip('.') for h in head_parameter_names(model, name)]
    peft_model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias='none',
        target_modules=LORA_TARGETS[name], modules_to_save=head))
    return peft_model
