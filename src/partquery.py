"""Decoder "part query" — encoder-decoder + cross-attention cho NABirds.

Kiến trúc
---------
    Encoder = backbone ĐÃ PRETRAIN (không viết lại, không train từ đầu)
        ViT-B/16 IN21k | BioCLIP | ConvNeXt-T IN22k
        -> token đặc trưng  [B, N, D]

    Decoder = K "part query" học được  [K, D],  L lớp:
        self-attn  giữa K query        -> các bộ phận chia việc cho nhau
        cross-attn K query -> N token  -> mỗi query hút về một vùng ảnh
        -> [B, K, D]

    Head = chiếu mỗi part xuống `proj_dim` rồi nối lại -> Linear(K*proj_dim, 555)

Vì sao KHÔNG đặt một query cho mỗi lớp (như INTR)
--------------------------------------------------
INTR (ICLR 2024) dùng C query, mỗi lớp một cái, và chính tác giả nêu hạn chế:
tốn khi C lớn hơn số ô lưới N. NABirds có **C = 555 > N = 196** (patch 16 ở ảnh
224) nên rơi đúng vùng xấu — và INTR chỉ đạt CUB 71.8% so với 83.8% của
ResNet-50 thường. Ở đây query gắn với **bộ phận**, K = 8..16 ≪ N, nên chi phí
decoder không đáng kể và không dính hạn chế đó.

Chống sập bộ phận
-----------------
Không ràng buộc gì thì mọi query sẽ hút về cùng một vùng. PDiscoFormer (ECCV
2024) dùng 6 loss; ở đây lấy hai cái rẻ và có tác dụng trực tiếp nhất, tính trên
bản đồ cross-attention trung bình của lớp cuối:
  - `orthogonality`: các bản đồ ít chồng nhau
  - `presence`: bộ phận nào cũng phải xuất hiện ở đâu đó trong ảnh

Chạy
----
    # rẻ nhất: đóng băng encoder, chỉ train decoder
    python src/partquery.py --model vit_b_16_in21k \\
        --encoder-ckpt runs/vit_b_16_in21k_224/best.pt --freeze-encoder

    # fine-tune cả hai, encoder khởi tạo từ checkpoint NABirds đã có
    python src/partquery.py --model vit_b_16_in21k \\
        --encoder-ckpt runs/vit_b_16_in21k_224/best.pt --epochs 15
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models_zoo
from metrics import compute_all, format_summary
from nabirds_data import build_loaders

REPO = models_zoo.REPO


# ---------------------------------------------------------------------------
# Encoder: lấy TOKEN chứ không lấy vector đã pool
# ---------------------------------------------------------------------------
class TokenEncoder(nn.Module):
    """Bọc backbone để trả về token [B, N, D] thay vì logits.

    timm ViT/Swin có `forward_features` trả [B, N, D] (ViT kèm cả class token,
    bỏ đi). ConvNeXt trả [B, C, H, W] -> duỗi thành [B, HW, C]. open_clip thì
    `visual` chỉ trả vector đã pool nên KHÔNG dùng làm encoder token được —
    chặn sớm thay vì để nó chạy sai lặng lẽ.
    """

    def __init__(self, name, ckpt=None, img_size=None):
        super().__init__()
        if name == 'bioclip':
            raise ValueError('bioclip tra ve vector da pool, khong co token map. '
                             'Dung vit_b_16_in21k / convnext_tiny_in22k.')
        model, self.img_size = models_zoo.build_model(name, img_size=img_size)
        if ckpt:
            sd = torch.load(ckpt, map_location='cpu')['model']
            missing, unexpected = model.load_state_dict(sd, strict=False)
            n = len(sd) - len(unexpected)
            if n < 0.5 * len(model.state_dict()):
                raise ValueError(f'chi nap {n} tensor tu {ckpt} -> khong khop kien truc')
            print(f'  encoder khoi tao tu {ckpt} ({n} tensor)', flush=True)
        self.name = name
        self.body = model
        self.norm_mean, self.norm_std = model.norm_mean, model.norm_std
        with torch.no_grad():
            t = self.forward(torch.zeros(1, 3, self.img_size, self.img_size))
        self.dim, self.n_tokens = t.shape[2], t.shape[1]

    def forward(self, x):
        f = self.body.forward_features(x)
        if f.ndim == 4:                      # ConvNeXt: [B, C, H, W]
            f = f.flatten(2).transpose(1, 2)
        elif self.name.startswith('vit'):    # timm ViT: bỏ class token
            f = f[:, 1:]
        return f


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, ffn_mult):
        super().__init__()
        self.n1, self.n2, self.n3 = (nn.LayerNorm(dim) for _ in range(3))
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        h = int(dim * ffn_mult)
        self.ffn = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, q, mem):
        h = self.n1(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.n2(q)
        out, attn = self.cross_attn(h, mem, mem, need_weights=True,
                                    average_attn_weights=True)
        q = q + out
        q = q + self.ffn(self.n3(q))
        return q, attn                       # attn: [B, K, N]


class PartQueryNet(nn.Module):
    def __init__(self, encoder, num_classes=models_zoo.NUM_CLASSES, k=12, layers=2,
                 heads=8, ffn_mult=2.0, proj_dim=128):
        super().__init__()
        self.encoder = encoder
        d = encoder.dim
        self.k = k
        self.query = nn.Parameter(torch.randn(1, k, d) * 0.02)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d, heads, ffn_mult) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, proj_dim)
        self.fc = nn.Linear(k * proj_dim, num_classes)
        nn.init.trunc_normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)
        self.last_attn = None

    def forward(self, x):
        mem = self.encoder(x)
        q = self.query.expand(mem.size(0), -1, -1)
        attn = None
        for blk in self.blocks:
            q, attn = blk(q, mem)
        self.last_attn = attn                # [B, K, N] của lớp cuối
        p = self.proj(self.norm(q))          # [B, K, proj_dim]
        return self.fc(p.flatten(1))


def part_losses(attn):
    """(orthogonality, presence) trên bản đồ cross-attention [B, K, N].

    orthogonality: chuẩn hoá từng bản đồ rồi phạt tích vô hướng giữa các cặp
      bộ phận khác nhau -> ép chúng nhìn vào những vùng khác nhau.
    presence: mỗi bộ phận phải có ít nhất một vị trí attention cao -> phạt
      (1 - max_n attn[k, n]) -> ép không có bộ phận nào bị bỏ trống.
    """
    a = F.normalize(attn, dim=2)                       # [B, K, N]
    gram = a @ a.transpose(1, 2)                       # [B, K, K]
    k = attn.size(1)
    off = gram - torch.diag_embed(torch.diagonal(gram, dim1=1, dim2=2))
    ortho = off.abs().sum((1, 2)) / max(1, k * (k - 1))
    presence = (1.0 - attn.max(dim=2).values).mean(1)
    return ortho.mean(), presence.mean()


@torch.no_grad()
def evaluate(model, loader, device, collect=False):
    model.eval()
    n = c1 = c5 = 0
    ys, ps, t5 = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x).float()
        top5 = out.topk(5, dim=1).indices
        hit5 = (top5 == y[:, None]).any(1)
        c1 += (top5[:, 0] == y).sum().item()
        c5 += hit5.sum().item()
        n += y.numel()
        if collect:
            ys.append(y.cpu()); ps.append(top5[:, 0].cpu()); t5.append(hit5.cpu())
    if not collect:
        return c1 / n, c5 / n, None
    return c1 / n, c5 / n, (torch.cat(ys).numpy(), torch.cat(ps).numpy(),
                            torch.cat(t5).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='vit_b_16_in21k')
    ap.add_argument('--encoder-ckpt', default=None,
                    help='checkpoint NABirds để khởi tạo encoder (rất nên dùng)')
    ap.add_argument('--freeze-encoder', action='store_true')
    ap.add_argument('--parts', type=int, default=12)
    ap.add_argument('--dec-layers', type=int, default=2)
    ap.add_argument('--proj-dim', type=int, default=128)
    ap.add_argument('--w-ortho', type=float, default=0.5)
    ap.add_argument('--w-presence', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=48)
    ap.add_argument('--img-size', type=int, default=None)
    ap.add_argument('--image-dir', default='images_r448')
    ap.add_argument('--lr-encoder', type=float, default=1e-4)
    ap.add_argument('--lr-decoder', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = 'cuda'

    enc = TokenEncoder(a.model, a.encoder_ckpt, a.img_size)
    net = PartQueryNet(enc, k=a.parts, layers=a.dec_layers,
                       proj_dim=a.proj_dim).to(device).to(memory_format=torch.channels_last)
    if a.freeze_encoder:
        for p in enc.parameters():
            p.requires_grad_(False)
    dec_params = [p for n, p in net.named_parameters()
                  if not n.startswith('encoder.') and p.requires_grad]
    enc_params = [p for n, p in net.named_parameters()
                  if n.startswith('encoder.') and p.requires_grad]
    n_dec = sum(p.numel() for p in dec_params)
    n_enc = sum(p.numel() for p in enc_params)

    run = a.run_name or (f'partq_{a.model}_{enc.img_size}_k{a.parts}'
                         + ('_frozen' if a.freeze_encoder else ''))
    run_dir, res_dir = os.path.join(REPO, 'runs', run), os.path.join(REPO, 'results', run)
    os.makedirs(run_dir, exist_ok=True)
    print(f'== {run} ==\n  encoder {a.model} dim {enc.dim} token {enc.n_tokens} '
          f'| decoder K={a.parts} L={a.dec_layers}\n  train duoc: decoder {n_dec/1e6:.2f}M '
          f'+ encoder {n_enc/1e6:.2f}M', flush=True)

    tr_set, va_set, te_set, tr_l, va_l, te_l = build_loaders(
        enc.img_size, a.batch_size, a.workers, False, 'standard', val_frac=0.15,
        seed=a.seed, image_dir=a.image_dir, mean=enc.norm_mean, std=enc.norm_std)

    groups = [{'params': dec_params, 'lr': a.lr_decoder, 'weight_decay': a.weight_decay}]
    if enc_params:
        groups.append({'params': enc_params, 'lr': a.lr_encoder,
                       'weight_decay': a.weight_decay})
    opt = torch.optim.AdamW(groups)
    steps = a.epochs * len(tr_l)
    warm = min(300, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / max(1, warm) if s < warm else
        0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm))))))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, best_ep, t0 = 0.0, 0, time.time()
    for ep in range(a.epochs):
        net.train()
        tot = corr = seen = 0
        for step, (x, y) in enumerate(tr_l):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = net(x)
                ce = crit(out, y)
                ortho, presence = part_losses(net.last_attn.float())
                loss = ce + a.w_ortho * ortho + a.w_presence * presence
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += ce.item() * y.numel()
            corr += (out.argmax(1) == y).sum().item()
            seen += y.numel()
            if a.max_steps and step + 1 >= a.max_steps:
                break
        acc1, acc5, _ = evaluate(net, va_l, device)
        star = ''
        if acc1 > best:
            best, best_ep, star = acc1, ep + 1, ' *'
            torch.save({'model': net.state_dict(), 'epoch': ep, 'top1': acc1,
                        'args': vars(a)}, os.path.join(run_dir, 'best.pt'))
        print(f'  ep {ep+1:02d}/{a.epochs}  ce {tot/seen:.3f}  ortho {ortho.item():.3f}  '
              f'pres {presence.item():.3f}  train-acc {corr/seen*100:5.2f}  |  '
              f'val {acc1*100:5.2f}{star}', flush=True)
    minutes = (time.time() - t0) / 60

    net.load_state_dict(torch.load(os.path.join(run_dir, 'best.pt'))['model'])
    _, _, packed = evaluate(net, te_l, device, collect=True)
    y_true, y_pred, t5 = packed
    summary, _ = compute_all(
        y_true, y_pred, t5, class_names=te_set.idx_to_name,
        species=te_set.idx_to_species, orders=te_set.idx_to_order,
        out_dir=res_dir, run_name=run,
        extra={'model': f'partq_{a.model}', 'img_size': enc.img_size,
               'params_M': round((n_dec + n_enc) / 1e6, 2), 'pretrained': True,
               'use_bbox': False, 'epochs': a.epochs, 'batch_size': a.batch_size,
               'train_minutes': round(minutes, 1), 'parts': a.parts,
               'dec_layers': a.dec_layers, 'frozen_encoder': a.freeze_encoder,
               'decoder_params_M': round(n_dec / 1e6, 2),
               'encoder_ckpt': a.encoder_ckpt, 'best_val_top1': best,
               'best_epoch': best_ep, 'epochs_run': a.epochs})
    print(format_summary(summary), flush=True)
    print(f'  -> {res_dir}', flush=True)


if __name__ == '__main__':
    main()
