"""Bước 3b (B2) — METS-analogue nhưng image tower được tinh chỉnh bằng LoRA.

Khác B1 (`train_mets.py`) đúng một điểm: B1 đóng băng image tower và chỉ học hai
projection head trên embedding đã cache; ở đây LoRA được gắn thêm vào image tower nên
đặc trưng ảnh cũng thay đổi được. Text tower vẫn ĐÓNG BĂNG hoàn toàn (tinh thần METS)
và prototype text vẫn precompute một lần.

Gotcha đã xác minh (README mục 9/10.4): open_clip dùng `nn.MultiheadAttention` với
q/k/v gộp trong `in_proj_weight` — KHÔNG phải `nn.Linear`, nên `peft` không gắn được
vào q/k/v. `nn.Linear` duy nhất trong `visual` là `c_fc`, `c_proj`, `out_proj`; LoRA
chỉ gắn vào ba module đó. Đo được: ViT-B/16 86.2M -> 1.77M tham số train được (2.05%).

    python zeroshot/src/train_lora.py --img bioclip --txt clip_b16 --level T0s --split variant
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

import zs_env  # noqa: E402
import zs_models  # noqa: E402
import prompts  # noqa: E402
import zs_eval  # noqa: E402
import splits as sp_mod  # noqa: E402
from train_mets import Aligner, gzsl_report, zsl_acc, text_prototypes  # noqa: E402
from nabirds_data import NABirds  # noqa: E402

TARGETS = ['c_fc', 'c_proj', 'out_proj']


def loader_for(labels_wanted, y_all, transform, batch, workers, split, shuffle):
    """DataLoader chỉ chứa ảnh của các lớp trong `labels_wanted`.

    `persistent_workers=True`: không có nó thì Windows spawn lại toàn bộ worker mỗi
    epoch và GPU tụt xuống ~3% utilisation — đúng sự cố đã ghi ở phần "Val loader
    spawn lại worker mỗi epoch" của README.
    """
    idx = np.where(np.isin(y_all, labels_wanted))[0]
    ds = NABirds(split=split, transform=transform, image_dir='images_r448',
                 indices=idx.tolist())
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
        pin_memory=True, drop_last=shuffle,
        persistent_workers=workers > 0), len(idx)


@torch.no_grad()
def encode_all(model, aligner, loader, device):
    """-> (embedding ảnh đã qua img_proj và L2-norm, nhãn lá)."""
    model.eval()
    zs, ys = [], []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for img, y in loader:
            f = model.encode_image(img.to(device, non_blocking=True)).float()
            zs.append(F.normalize(aligner.img_proj(f), dim=-1).cpu())
            ys.append(y)
    return torch.cat(zs), torch.cat(ys).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', default='bioclip', choices=list(zs_models.REGISTRY))
    ap.add_argument('--txt', default='clip_b16', choices=list(zs_models.REGISTRY))
    ap.add_argument('--level', default='T0s')
    ap.add_argument('--split', default='variant', choices=['variant', 'species'])
    ap.add_argument('--dim', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-4)      # LoRA
    ap.add_argument('--head-lr', type=float, default=1e-3)  # projection head
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--val-class-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    torch.manual_seed(args.seed)
    device = 'cuda'
    run = f'lora_{args.img}IMG_{args.txt}TXT_{args.level}_{args.split}'
    print(run, flush=True)

    model, pre_val, _ = zs_models.load(args.img, device)
    with torch.no_grad():                                  # chiều embedding ảnh
        d_img = model.encode_image(torch.zeros(1, 3, 224, 224, device=device)).shape[1]

    # Transform train = 3 bước cuối của preprocess gốc (convert / to-tensor /
    # normalize theo chuẩn CLIP), thay Resize+CenterCrop bằng RandomResizedCrop + lật.
    # scale=(0.5, 1.0) chứ không phải (0.08, 1.0) mặc định: bbox con chim chỉ chiếm
    # median 28.3% diện tích ảnh nên crop mạnh là cắt mất con chim (README mục 0).
    from torchvision import transforms as T
    pre_train = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0),
                                               interpolation=T.InterpolationMode.BICUBIC),
                           T.RandomHorizontalFlip(0.5)] + list(pre_val.transforms[2:]))

    for p in model.parameters():
        p.requires_grad = False
    model.visual = get_peft_model(model.visual, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias='none',
        target_modules=TARGETS))
    n_lora = sum(p.numel() for p in model.visual.parameters() if p.requires_grad)

    txt = text_prototypes(args.txt, args.level).to(device)
    unseen = np.load(os.path.join(zs_env.DATA_DIR, 'splits.npz'))[args.split]
    seen_idx = np.where(~unseen)[0]

    rng = np.random.default_rng(args.seed)
    n_hold = max(1, int(round(args.val_class_frac * len(seen_idx))))
    val_cls = set(rng.choice(seen_idx, size=n_hold, replace=False).tolist())
    fit_idx = np.array([c for c in seen_idx if c not in val_cls])
    val_idx = np.array(sorted(val_cls))
    fit_remap = -np.ones(555, dtype=np.int64)
    fit_remap[fit_idx] = np.arange(len(fit_idx))

    y_train_all = NABirds(split='train').samples
    y_train_all = np.array([s[2] for s in y_train_all])
    fit_loader, n_fit = loader_for(fit_idx, y_train_all, pre_train, args.batch_size,
                                   args.workers, 'train', True)
    val_loader, n_val = loader_for(val_idx, y_train_all, pre_val, args.batch_size,
                                   args.workers, 'train', False)
    test_ds = NABirds(split='test', transform=pre_val, image_dir='images_r448')
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print(f'  LoRA {n_lora/1e6:.2f}M | fit {n_fit} anh / {len(fit_idx)} lop | '
          f'val-unseen {n_val} anh / {len(val_idx)} lop', flush=True)

    aligner = Aligner(d_img, txt.shape[1], args.dim).to(device)
    opt = torch.optim.AdamW(
        [{'params': [p for p in model.visual.parameters() if p.requires_grad],
          'lr': args.lr},
         {'params': aligner.parameters(), 'lr': args.head_lr}], weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(fit_loader))
    txt_fit, txt_val = txt[fit_idx], txt[val_idx]
    val_remap = {c: i for i, c in enumerate(val_idx)}

    best_val, best_state, t0 = -1, None, time.time()
    for ep in range(args.epochs):
        model.train()
        tot = n = 0
        for img, y in fit_loader:
            img = img.to(device, non_blocking=True)
            yb = torch.from_numpy(fit_remap[y.numpy()]).to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = model.encode_image(img)
            loss = F.cross_entropy(aligner(f.float(), aligner.protos(txt_fit)), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
            n += len(yb)

        z, yv = encode_all(model, aligner, val_loader, device)
        pv = aligner.protos(txt_val).detach().cpu()
        pred = (z @ pv.T).argmax(1).numpy()
        va = float(np.mean(pred == np.array([val_remap[c] for c in yv])))
        if va > best_val:
            best_val = va
            best_state = ({k: v.detach().clone() for k, v in
                           model.visual.state_dict().items() if 'lora' in k},
                          {k: v.detach().clone() for k, v in aligner.state_dict().items()})
        print(f'  ep{ep} loss {tot/n:.4f}  val-UNSEEN {100*va:.2f}  '
              f'[{time.time()-t0:.0f}s, VRAM {torch.cuda.max_memory_allocated()/2**30:.1f}GB]',
              flush=True)

    model.visual.load_state_dict(best_state[0], strict=False)
    aligner.load_state_dict(best_state[1])
    z, yte = encode_all(model, aligner, test_loader, device)
    logits = (z @ aligner.protos(txt).detach().cpu().T).numpy()

    l2s = zs_eval.leaf_to_species_index(prompts.load_tables())
    sub81 = set(l2s[np.where(sp_mod.variant_split(prompts.load_tables()))[0]].tolist())
    res = {'run': run, 'img': args.img, 'txt': args.txt, 'level': args.level,
           'split': args.split, 'mode': 'lora', 'rank': args.rank,
           'lora_params_M': n_lora / 1e6, 'epochs': args.epochs,
           'peak_vram_GB': torch.cuda.max_memory_allocated() / 2**30,
           'minutes': (time.time() - t0) / 60, 'val_unseen_acc': best_val,
           'zsl_unseen_acc': zsl_acc(logits, yte, unseen),
           **gzsl_report(logits, yte, unseen),
           'all555_top1': float((logits.argmax(1) == yte).mean()),
           **zs_eval.variant_probe(logits, yte, l2s),
           **zs_eval.variant_probe(logits, yte, l2s, sub81, '81')}

    out = os.path.join(zs_env.RESULTS_DIR, run)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print('  LORA ' + ' '.join(
        f'{k}={100*res[k]:.2f}' for k in
        ('zsl_unseen_acc', 'gzsl_harmonic', 'vp_bal', 'vp81_bal', 'all555_top1')))
    print(f'  VRAM {res["peak_vram_GB"]:.1f}GB, {res["minutes"]:.1f} phut -> {out}')


if __name__ == '__main__':
    main()
