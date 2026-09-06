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


def _vit_b16_in21k(pretrained):
    """ViT-B/16 pretrain ImageNet-21k (timm) — đây là checkpoint mà TransFG và
    các paper FGVC dùng, mạnh hơn hẳn bản ImageNet-1k của torchvision."""
    import timm
    m = timm.create_model('vit_base_patch16_224.augreg_in21k',
                          pretrained=pretrained, num_classes=NUM_CLASSES)
    return m, 224


def _inception_v3(pretrained):
    """Inception-v3 dùng input 299x299 và có nhánh phụ AuxLogits.

    torchvision ép transform_input=True khi load trọng số pretrain (trọng số
    port từ TensorFlow), nên vẫn dùng chuẩn hoá ImageNet bình thường ở dataset.
    """
    w = tvm.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.inception_v3(weights=w, aux_logits=True, init_weights=not pretrained)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, NUM_CLASSES)
    return m, 299


BUILDERS = {
    'cnn_scratch':   lambda pretrained: (SimpleCNN(), 224),
    'resnet50':      lambda pretrained: _resnet(50, pretrained),
    'resnet101':     lambda pretrained: _resnet(101, pretrained),
    'vit_b_16':      _vit_b16,
    'vit_b_16_in21k': _vit_b16_in21k,
    'inception_v3':  _inception_v3,
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


def build_model(name, pretrained=True):
    if name not in BUILDERS:
        raise ValueError(f'model phải thuộc {MODEL_NAMES}, nhận được {name!r}')
    if name == 'cnn_scratch':
        pretrained = False
    model, img_size = BUILDERS[name](pretrained)
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
    return model, img_size


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
    return ['fc.']


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
