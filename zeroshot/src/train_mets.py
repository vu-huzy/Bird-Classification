"""Bước 3 — METS-analogue: text tower ĐÓNG BĂNG, chỉ train phía ảnh + 2 projection head.

    ảnh  --[image tower]--> D_img --[img_proj, TRAIN]--> d --L2norm--\
                                                                      >-- logits = cos/tau
    text --[text tower, ĐÓNG BĂNG]--> D_txt --[txt_proj, TRAIN]--> d --L2norm--/

Đúng tinh thần METS (MIDL'23): LM đóng băng, mỗi nhánh một linear head train được.
Khác một điểm phải nói rõ trong báo cáo: METS có text MỨC INSTANCE (mỗi ECG một báo
cáo riêng), NABirds chỉ có text MỨC LỚP. Với text mức lớp, InfoNCE in-batch sinh
false negative (ảnh cùng lớp trong batch bị đẩy ra xa nhau), nên ở đây bỏ hẳn
in-batch negative: mỗi step so ảnh với TOÀN BỘ prototype của các lớp seen.
Về mặt toán học đây là cross-entropy với classifier head sinh từ embedding text —
tức dòng DeViSE/ALE/ESZSL, không phải CLIP. (README mục 3.2)

Hệ quả thực dụng: số negative = số lớp seen, không phụ thuộc batch size -> KHÔNG cần
batch lớn, KHÔNG cần gradient accumulation.

    python zeroshot/src/train_mets.py --img bioclip --txt bioclip --level T2 --split variant
    python zeroshot/src/train_mets.py --img bioclip --txt clip_b16 --level T0v --split variant
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

import zs_env  # noqa: E402
import zs_models  # noqa: E402
import prompts  # noqa: E402
import encode as enc  # noqa: E402
import zs_eval  # noqa: E402

TXT_CACHE = os.path.join(zs_env.CACHE_DIR, 'txt')


def text_prototypes(txt_model, level, taxo=None, device='cuda'):
    """(555, D_txt) đã L2-norm. Cache lại vì encode 555x80 câu mất ~20s."""
    os.makedirs(TXT_CACHE, exist_ok=True)
    taxo = taxo or ('tol' if 'bioclip' in txt_model else 'gbif')
    p = os.path.join(TXT_CACHE, f'{txt_model}_{level}_{taxo}_leaf.npy')
    if os.path.exists(p):
        return torch.from_numpy(np.load(p))
    df = prompts.load_tables()
    texts, _ = prompts.class_texts(df, level, space='leaf', taxo=taxo)
    model, _, tok = zs_models.load(txt_model, device)
    tf = zs_eval.text_features(model, tok, texts, device).cpu()
    del model
    torch.cuda.empty_cache()
    np.save(p, tf.numpy())
    return tf


class Aligner(nn.Module):
    """Hai linear head (METS) + nhiệt độ học được (CLIP)."""

    def __init__(self, d_img, d_txt, d=512):
        super().__init__()
        self.img_proj = nn.Linear(d_img, d, bias=False)
        self.txt_proj = nn.Linear(d_txt, d, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def protos(self, txt):
        return F.normalize(self.txt_proj(txt), dim=-1)

    def forward(self, img, protos):
        z = F.normalize(self.img_proj(img), dim=-1)
        return self.logit_scale.clamp(max=np.log(100)).exp() * z @ protos.T


def harmonic(a, b):
    return 0.0 if (a + b) == 0 else 2 * a * b / (a + b)


def gzsl_report(logits_all, y, unseen_mask):
    """logits_all: (N, 555) trên TOÀN BỘ 555 lớp -> acc seen / unseen / harmonic."""
    pred = logits_all.argmax(1)
    is_unseen = unseen_mask[y]
    accs = {}
    for tag, m in (('seen', ~is_unseen), ('unseen', is_unseen)):
        accs[f'gzsl_{tag}'] = float((pred[m] == y[m]).mean()) if m.sum() else 0.0
    accs['gzsl_harmonic'] = harmonic(accs['gzsl_seen'], accs['gzsl_unseen'])
    return accs


def zsl_acc(logits_all, y, unseen_mask):
    """ZSL thuần: chỉ ảnh của lớp unseen, chỉ chọn trong các lớp unseen."""
    idx = np.where(unseen_mask)[0]
    m = unseen_mask[y]
    sub = logits_all[m][:, idx]
    remap = {c: i for i, c in enumerate(idx)}
    tgt = np.array([remap[c] for c in y[m]])
    return float((sub.argmax(1) == tgt).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', default='bioclip', choices=list(zs_models.REGISTRY))
    ap.add_argument('--txt', default='bioclip', choices=list(zs_models.REGISTRY))
    ap.add_argument('--level', default='T2')
    ap.add_argument('--split', default='variant', choices=['variant', 'species'])
    ap.add_argument('--dim', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--val-class-frac', type=float, default=0.15,
                    help='ti le LOP seen giu lai lam unseen gia de chon epoch')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda'
    run = (f'mets_{args.img}IMG_{args.txt}TXT_{args.level}_{args.split}'
           + (f'_s{args.seed}' if args.seed else ''))

    xtr = torch.from_numpy(np.load(enc.paths(args.img, 'train')[0]))
    ytr = np.load(enc.paths(args.img, 'train')[1])
    xte = torch.from_numpy(np.load(enc.paths(args.img, 'test')[0]))
    yte = np.load(enc.paths(args.img, 'test')[1])
    txt = text_prototypes(args.txt, args.level).to(device)
    unseen = np.load(os.path.join(zs_env.DATA_DIR, 'splits.npz'))[args.split]
    seen_idx = np.where(~unseen)[0]
    remap = -np.ones(555, dtype=np.int64)
    remap[seen_idx] = np.arange(len(seen_idx))

    # --- chọn epoch bằng VAL-UNSEEN, không phải val-seen (Xian et al.) ---------
    # Chọn theo val của lớp seen chính là tối đa hoá thứ đối nghịch với zero-shot:
    # đo thử cách đó cho ZSL-unseen 62.59% trong khi không train gì được 69.70%.
    # Vì vậy cắt tiếp một phần LỚP seen ra làm tập unseen giả để chọn epoch.
    rng = np.random.default_rng(args.seed)
    n_hold = max(1, int(round(args.val_class_frac * len(seen_idx))))
    val_cls = set(rng.choice(seen_idx, size=n_hold, replace=False).tolist())
    fit_idx = np.array([c for c in seen_idx if c not in val_cls])
    val_idx = np.array(sorted(val_cls))
    fit_remap = -np.ones(555, dtype=np.int64)
    fit_remap[fit_idx] = np.arange(len(fit_idx))

    tr_keep = np.isin(ytr, fit_idx)                      # train: lớp seen trừ val-class
    va_keep = np.isin(ytr, val_idx)                      # val  : các lớp bị giữ lại
    xfit = xtr[tr_keep].to(device)
    yfit = torch.from_numpy(fit_remap[ytr[tr_keep]]).to(device)
    xval, yval = xtr[va_keep].to(device), ytr[va_keep]
    print(f'{run}\n  seen {len(seen_idx)} lop -> fit {len(fit_idx)} / val-unseen {len(val_idx)}')
    print(f'  train {tuple(xfit.shape)} | val {tuple(xval.shape)} | '
          f'test {tuple(xte.shape)} | unseen that {int(unseen.sum())} lop')

    model = Aligner(xtr.shape[1], txt.shape[1], args.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    txt_fit, txt_val = txt[fit_idx], txt[val_idx]
    val_remap = {c: i for i, c in enumerate(val_idx)}
    yval_t = torch.tensor([val_remap[c] for c in yval], device=device)

    best_val, best_state = -1, None
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(len(xfit), device=device)
        tot = 0.0
        for i in range(0, len(order), args.batch_size):
            b = order[i:i + args.batch_size]
            loss = F.cross_entropy(model(xfit[b], model.protos(txt_fit)), yfit[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():                            # ZSL trên các lớp val giữ lại
            va = (model(xval, model.protos(txt_val)).argmax(1) == yval_t).float().mean().item()
        if va > best_val:
            best_val, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f'  ep{ep:3d} loss {tot/len(order):.4f}  val-UNSEEN {100*va:.2f}')

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        protos_all = model.protos(txt)
        logits = model(xte.to(device), protos_all).cpu().numpy()

    l2s = zs_eval.leaf_to_species_index(prompts.load_tables())
    import splits as sp_mod
    sub81 = set(l2s[np.where(sp_mod.variant_split(prompts.load_tables()))[0]].tolist())
    res = {'run': run, 'img': args.img, 'txt': args.txt, 'level': args.level,
           'split': args.split, 'mode': 'linear', 'dim': args.dim,
           'seed': args.seed,
           'val_unseen_acc': best_val,
           'zsl_unseen_acc': zsl_acc(logits, yte, unseen),
           **gzsl_report(logits, yte, unseen),
           'all555_top1': float((logits.argmax(1) == yte).mean()),
           **zs_eval.oracle_variant_acc(logits, yte, l2s),
           **zs_eval.variant_probe(logits, yte, l2s),
           **zs_eval.variant_probe(logits, yte, l2s, sub81, '81')}

    # Mốc B0: đúng cùng giao thức nhưng KHÔNG train gì (chỉ dùng khi 2 tower cùng model)
    if args.img == args.txt:
        base = (xte.to(device) @ txt.T).cpu().numpy()
        res.update({'b0_zsl_unseen_acc': zsl_acc(base, yte, unseen),
                    **{f'b0_{k}': v for k, v in gzsl_report(base, yte, unseen).items()},
                    'b0_all555_top1': float((base.argmax(1) == yte).mean()),
                    **{f'b0_{k}': v for k, v in
                       zs_eval.oracle_variant_acc(base, yte, l2s).items()},
                    **{f'b0_{k}': v for k, v in
                       zs_eval.variant_probe(base, yte, l2s).items()},
                    **{f'b0_{k}': v for k, v in
                       zs_eval.variant_probe(base, yte, l2s, sub81, '81').items()}})

    out = os.path.join(zs_env.RESULTS_DIR, run)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    show = ('zsl_unseen_acc', 'gzsl_harmonic', 'vp_bal', 'vp81_bal', 'all555_top1')
    print('  TRAINED ' + ' '.join(f'{k}={100*res[k]:.2f}' for k in show if k in res))
    if 'b0_zsl_unseen_acc' in res:
        print('  B0      ' + ' '.join(f'{k}={100*res["b0_" + k]:.2f}'
                                      for k in show if 'b0_' + k in res))
    print(f'  -> {out}/summary.json')


if __name__ == '__main__':
    main()
