"""Bước 3c (B4) — contrastive THẬT trên CUB (ảnh, caption), rồi zero-shot sang NABirds.

Đây là biến thể trung thành với METS nhất trong cả dự án, và là phép đo trực tiếp cho
luận điểm ở mục 3.2. Biến độc lập duy nhất: **độ hạt của text**.

  `--caps naive` : 200 chuỗi duy nhất / 5,994 ảnh -> text MỨC LỚP.
                   Trong một batch, mọi ảnh cùng loài nhận CÙNG một chuỗi -> InfoNCE
                   phạt model vì kéo chúng lại gần nhau (false negative).
  `--caps cupl`  : 4,490 chuỗi duy nhất / 5,994 ảnh -> text BIẾN THIÊN theo ảnh,
                   đúng cấu trúc (ECG, báo cáo máy sinh) của METS.

Khác B1/B2: ở đây dùng InfoNCE đối xứng in-batch THẬT (loss của CLIP), không phải
cross-entropy trên prototype lớp — vì bây giờ text mới đủ đa dạng để chuyện đó có nghĩa.

Đánh giá: NABirds, **352 lá có loài KHÔNG nằm trong 200 lớp CUB** (14,907 ảnh test).
Prototype NABirds dựng bằng chính text tower đóng băng + `txt_proj` đã học, dùng cùng
đường ensemble 80 template như mọi bảng khác nên số liệu so trực tiếp được.

    python zeroshot/src/train_cub.py --img bioclip --txt clip_b16 --caps cupl
"""
import argparse
import io
import json
import os
import sys

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
import build_cub  # noqa: E402
import encode as enc  # noqa: E402
from train_mets import Aligner, text_prototypes  # noqa: E402

CUB_CACHE = os.path.join(zs_env.CACHE_DIR, 'cub')


def cub_features(img_model, txt_model, caps, device, batch=64):
    """Embedding ảnh CUB (image tower) + embedding caption (text tower), đều đóng băng."""
    os.makedirs(CUB_CACHE, exist_ok=True)
    fi = os.path.join(CUB_CACHE, f'img_{img_model}_{caps}.npy')
    ft = os.path.join(CUB_CACHE, f'txt_{txt_model}_{caps}.npy')
    if os.path.exists(fi) and os.path.exists(ft):
        return torch.from_numpy(np.load(fi)), torch.from_numpy(np.load(ft))

    from PIL import Image
    d = build_cub.load_split(caps)
    model, preprocess, tok = zs_models.load(img_model, device)
    feats = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(d), batch):
            ims = torch.stack([preprocess(Image.open(io.BytesIO(b)).convert('RGB'))
                               for b in d.image_bytes.iloc[i:i + batch]]).to(device)
            feats.append(F.normalize(model.encode_image(ims), dim=-1).float().cpu())
    img = torch.cat(feats)
    del model
    torch.cuda.empty_cache()

    tmodel, _, ttok = zs_models.load(txt_model, device)
    outs = []
    texts = d.text.tolist()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(texts), 256):
            outs.append(F.normalize(
                tmodel.encode_text(ttok(texts[i:i + 256]).to(device)), dim=-1).float().cpu())
    txt = torch.cat(outs)
    del tmodel
    torch.cuda.empty_cache()

    np.save(fi, img.numpy())
    np.save(ft, txt.numpy())
    return img, txt


def cub_labels(caps, names):
    """Nhãn lớp CUB cho từng ảnh. Cần để tách VAL THEO LỚP (chọn epoch kiểu ZSL).

    `naive`: caption chính là tên lớp -> khớp 100%.
    `cupl` : caption có nhắc tên loài ("A Laysan albatross is a seabird...") -> tìm tên
             lớp DÀI NHẤT xuất hiện trong caption; khớp 5,365/5,994 = 89.5%, phủ đủ
             200 lớp. Ảnh không khớp (-1) vẫn dùng để train, chỉ không dùng làm val.
    """
    d = build_cub.load_split(caps)
    order = sorted(names, key=len, reverse=True)
    idx = {' '.join(build_cub._norm(n)): i for i, n in enumerate(names)}
    out = []
    for t in d.text:
        nt = ' '.join(build_cub._norm(t))
        out.append(next((idx[' '.join(build_cub._norm(n))] for n in order
                         if ' '.join(build_cub._norm(n)) in nt), -1))
    return np.array(out)


def cub_class_protos(txt_model, names, device):
    """(200, D) prototype tên lớp CUB, cùng đường ensemble 80 template như mọi bảng khác."""
    p = os.path.join(CUB_CACHE, f'proto_{txt_model}.npy')
    if os.path.exists(p):
        return torch.from_numpy(np.load(p))
    model, _, tok = zs_models.load(txt_model, device)
    f = zs_eval.text_features(model, tok, list(names), device).cpu()
    del model
    torch.cuda.empty_cache()
    np.save(p, f.numpy())
    return f


def infonce(z_img, z_txt, scale):
    """InfoNCE đối xứng của CLIP. Nhãn = đường chéo (ghép cặp theo vị trí trong batch)."""
    logits = scale * z_img @ z_txt.T
    tgt = torch.arange(len(z_img), device=z_img.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', default='bioclip', choices=list(zs_models.REGISTRY))
    ap.add_argument('--txt', default='clip_b16', choices=list(zs_models.REGISTRY))
    ap.add_argument('--caps', default='cupl', choices=['naive', 'cupl'])
    ap.add_argument('--level', default='T0s', help='prompt dung cho prototype NABirds')
    ap.add_argument('--dim', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda'
    run = f'cub_{args.img}IMG_{args.txt}TXT_{args.caps}_lr{args.lr:g}'
    print(run, flush=True)

    ci, ct = cub_features(args.img, args.txt, args.caps, device)
    ci, ct = ci.to(device), ct.to(device)
    n_uniq = len(set(build_cub.load_split(args.caps).text))
    print(f'  CUB {tuple(ci.shape)} anh | {n_uniq} chuoi text duy nhat '
          f'({n_uniq/len(ci):.2f}/anh)', flush=True)

    # --- VAL THEO LỚP: giữ 20% lớp CUB khỏi train, chọn epoch bằng ZSL trên chúng ---
    # Không có bước này thì InfoNCE nhớ thuộc 5,994 cặp (loss 0.11) và phá sạch khả năng
    # tổng quát hoá: đo được zsl352 = 4.86% so với 44.90% khi không train gì.
    names = build_cub.cub_class_names(build_cub.load_split('naive'))
    lab = cub_labels(args.caps, names)
    rng = np.random.default_rng(args.seed)
    val_cls = set(rng.choice(200, size=40, replace=False).tolist())
    is_val = np.isin(lab, list(val_cls))
    fit_i = torch.from_numpy(np.where(~is_val)[0]).to(device)
    val_i = np.where(is_val)[0]
    protos200 = cub_class_protos(args.txt, names, device).to(device)
    val_idx = np.array(sorted(val_cls))
    val_tgt = torch.from_numpy(
        np.searchsorted(val_idx, lab[val_i])).to(device)
    ci_val = ci[torch.from_numpy(val_i).to(device)]
    print(f'  train {len(fit_i)} anh / 160 lop | val-unseen {len(val_i)} anh / 40 lop',
          flush=True)

    aligner = Aligner(ci.shape[1], ct.shape[1], args.dim).to(device)
    opt = torch.optim.AdamW(aligner.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_val, best_state = -1, None
    for ep in range(args.epochs):
        perm = fit_i[torch.randperm(len(fit_i), device=device)]
        tot = nb = 0
        for i in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            b = perm[i:i + args.batch_size]
            zi = F.normalize(aligner.img_proj(ci[b]), dim=-1)
            zt = F.normalize(aligner.txt_proj(ct[b]), dim=-1)
            loss = infonce(zi, zt, aligner.logit_scale.clamp(max=np.log(100)).exp())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        with torch.no_grad():
            zv = F.normalize(aligner.img_proj(ci_val), dim=-1)
            pv = aligner.protos(protos200[torch.from_numpy(val_idx).to(device)])
            va = float(((zv @ pv.T).argmax(1) == val_tgt).float().mean())
        if va > best_val:
            best_val, best_state = va, {k: v.clone() for k, v in aligner.state_dict().items()}
        if ep % 15 == 0 or ep == args.epochs - 1:
            print(f'  ep{ep:3d} infonce {tot/max(1, nb):.4f}  val-UNSEEN(40 lop CUB) '
                  f'{100*va:.2f}', flush=True)
    aligner.load_state_dict(best_state)

    # ---- zero-shot sang NABirds, chi cac la co loai KHONG nam trong CUB ----
    aligner.eval()
    xte = torch.from_numpy(np.load(enc.paths(args.img, 'test')[0])).to(device)
    yte = np.load(enc.paths(args.img, 'test')[1])
    txt = text_prototypes(args.txt, args.level).to(device)
    unseen = np.load(os.path.join(zs_env.DATA_DIR, 'cub_unseen_leaf.npy'))

    with torch.no_grad():
        z = F.normalize(aligner.img_proj(xte), dim=-1)
        logits = (z @ aligner.protos(txt).T).cpu().numpy()
        base = (xte @ txt.T).cpu().numpy()            # B0: khong train gi

    l2s = zs_eval.leaf_to_species_index(prompts.load_tables())
    idx = np.where(unseen)[0]
    m = unseen[yte]
    remap = {c: i for i, c in enumerate(idx)}
    tgt = np.array([remap[c] for c in yte[m]])
    sub_un = set(l2s[idx].tolist())

    def pack(lg, pre=''):
        return {f'{pre}zsl352_top1': float((lg[m][:, idx].argmax(1) == tgt).mean()),
                f'{pre}all555_top1': float((lg.argmax(1) == yte).mean()),
                **{f'{pre}{k}': v for k, v in
                   zs_eval.variant_probe(lg, yte, l2s, sub_un, 'un').items()}}

    res = {'run': run, 'img': args.img, 'txt': args.txt, 'caps': args.caps,
           'level': args.level, 'mode': 'cub_infonce', 'n_cub': int(len(ci)),
           'lr': args.lr,
           'uniq_texts': n_uniq, 'texts_per_img': n_uniq / len(ci),
           'n_unseen_leaves': int(unseen.sum()), 'n_unseen_test': int(m.sum()),
           'cub_val_unseen_acc': best_val,
           **pack(logits), **pack(base, 'b0_')}

    out = os.path.join(zs_env.RESULTS_DIR, run)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    keys = ('zsl352_top1', 'all555_top1', 'vpun_bal')
    print('  CUB-TRAINED ' + ' '.join(f'{k}={100*res[k]:.2f}' for k in keys if k in res))
    print('  B0          ' + ' '.join(f'{k}={100*res["b0_" + k]:.2f}'
                                      for k in keys if 'b0_' + k in res))
    print(f'  -> {out}/summary.json')


if __name__ == '__main__':
    main()
