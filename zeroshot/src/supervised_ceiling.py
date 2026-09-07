"""Trần tham chiếu: model CÓ GIÁM SÁT đạt bao nhiêu trên chính trục biến thể?

Mọi con số zero-shot ở bước 2/3 đều cần một mốc trên để đọc. `vit_b_16_in21k` của bài
555 lớp (top-1 85.86%) đã thấy đủ 555 nhãn khi train, kể cả nhãn biến thể — nên
`vp81_bal` của nó chính là trần thực nghiệm cho phần "đọc bộ lông" trên dataset này.

Chạy: `python zeroshot/src/supervised_ceiling.py --run vit_b_16_in21k_224`
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

import zs_env  # noqa: E402
import prompts  # noqa: E402
import zs_eval  # noqa: E402
import splits as sp_mod  # noqa: E402
from models_zoo import build_model  # noqa: E402
from nabirds_data import NABirds, build_transforms  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='vit_b_16_in21k_224')
    ap.add_argument('--model', default='vit_b_16_in21k')
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--lora-rank', type=int, default=0,
                    help='dat >0 neu checkpoint duoc train bang --lora-rank')
    args = ap.parse_args()

    ck = torch.load(os.path.join(REPO, 'runs', args.run, 'best.pt'),
                    map_location='cpu', weights_only=False)
    model, _ = build_model(args.model, pretrained=False)
    if args.lora_rank:                                   # key khac nhau -> phai boc LoRA truoc
        from models_zoo import apply_lora
        model = apply_lora(model, args.model, args.lora_rank)
    model.load_state_dict(ck['model'])
    model = model.cuda().eval()
    print(f'{args.run}: epoch {ck["epoch"]}, val top1 {100*ck["top1"]:.2f}')

    ds = NABirds(split='test', transform=build_transforms(args.img_size, train=False),
                 image_dir='images_r448')
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)
    logits, ys = [], []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for x, y in loader:
            logits.append(model(x.cuda(non_blocking=True)).float().cpu())
            ys.append(y)
    logits = torch.cat(logits).numpy()
    y = torch.cat(ys).numpy()

    df = prompts.load_tables()
    l2s = zs_eval.leaf_to_species_index(df)
    sub81 = set(l2s[np.where(sp_mod.variant_split(df))[0]].tolist())
    res = {'run': f'supervised_{args.run}', 'mode': 'supervised',
           'lora_rank': args.lora_rank,
           'top1': float((logits.argmax(1) == y).mean()),
           **zs_eval.variant_probe(logits, y, l2s),
           **zs_eval.variant_probe(logits, y, l2s, sub81, '81')}

    out = os.path.join(zs_env.RESULTS_DIR, res['run'])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(f'  top1(555)  = {100*res["top1"]:.2f}   (bang ket qua da co: 85.86)')
    print(f'  vp_bal     = {100*res["vp_bal"]:.2f}   (137 loai da-la)')
    print(f'  vp81_bal   = {100*res["vp81_bal"]:.2f}   (81 loai cua split variant)')
    print(f'  chance     = {100*res["vp_chance"]:.2f}')
    print(f'  -> {out}/summary.json')


if __name__ == '__main__':
    main()
