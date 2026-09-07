"""Ba checkpoint CLIP-style dùng trong thí nghiệm, nạp qua open_clip.

| tên       | kiến trúc | embed | pretrain                    | vai trò |
|-----------|-----------|-------|-----------------------------|---------|
| clip_b16  | ViT-B/16  | 512   | OpenAI WIT-400M             | mốc domain-generic |
| bioclip   | ViT-B/16  | 512   | TreeOfLife-10M              | domain-matched, CÙNG CỠ clip_b16 -> so sánh sạch |
| bioclip2  | ViT-L/14  | 768   | TreeOfLife-200M             | domain-matched, mạnh hơn nhưng khác cỡ |

Cả ba dùng chuẩn hoá CLIP (mean 0.4815/0.4578/0.4082), KHÁC chuẩn hoá ImageNet mà
`src/nabirds_data.py` dùng -> luôn lấy `preprocess` trả về từ đây, đừng tái dụng
transform của bài 555 lớp.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zs_env  # noqa: F401,E402  (đặt HF_HOME + USE_TF trước khi nạp model)

# `ViT-B-16-quickgelu` chứ KHÔNG phải `ViT-B-16`: CLIP gốc của OpenAI dùng QuickGELU.
# Nạp bằng `ViT-B-16` + tag `openai` thì open_clip cảnh báo
# "QuickGELU mismatch ... (quick_gelu=False) and pretrained tag 'openai' (quick_gelu=True)"
# và dựng model SAI activation -> mọi số của clip_b16 bị hạ oan.
REGISTRY = {
    'clip_b16': ('ViT-B-16-quickgelu', 'openai'),
    'bioclip': ('hf-hub:imageomics/bioclip', None),
    'bioclip2': ('hf-hub:imageomics/bioclip-2', None),
}


def load(name, device='cuda'):
    """-> (model ở chế độ eval, preprocess cho ảnh, tokenizer)."""
    import open_clip
    arch, pretrained = REGISTRY[name]
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained, cache_dir=os.path.join(zs_env.MODEL_DIR, 'openclip'))
    tokenizer = open_clip.get_tokenizer(arch)
    return model.to(device).eval(), preprocess, tokenizer
