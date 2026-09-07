"""Ensemble + TTA từ các checkpoint ĐÃ CÓ — không train thêm gì.

Hai đòn bẩy miễn phí mà repo chưa dùng:
  TTA       eval thêm bản lật ngang, và (với model toàn tích chập) eval ở độ phân
            giải cao hơn lúc train — hiệu ứng "train-test resolution discrepancy".
  Ensemble  trung bình xác suất softmax của nhiều checkpoint. ViT và CNN sai KHÁC
            kiểu nhau (err_same_order 92.3 vs 86.1) nên gộp lại thường có lãi.

QUAN TRỌNG — chọn cấu hình bằng VAL, báo cáo trên TEST:
  Chọn TTA tốt nhất và chọn thành viên ensemble bằng chính test là gian lận, cùng
  loại lỗi với bẫy #11 của repo. Mọi run đều dùng `--val-frac 0.15 --seed 0` nên
  `stratified_val_split` sinh ra CÙNG MỘT tập val -> so sánh trên val là hợp lệ.

  python src/ensemble.py --cache      # tính & cache logits val+test cho mọi run
  python src/ensemble.py --report     # bảng TTA + ensemble chọn tham lam trên val
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import models_zoo
from metrics import compute_all
from nabirds_data import NABirds, build_transforms, stratified_val_split

REPO = models_zoo.REPO
RESULTS = os.path.join(REPO, 'results')
RUNS = os.path.join(REPO, 'runs')

# Model chấp nhận input kích thước tuỳ ý (toàn tích chập + adaptive pool). ViT và
# Swin thì không: pos-embed / window size gắn chặt với 224.
VARIABLE_INPUT = {'cnn_scratch', 'resnet50', 'resnet101', 'inception_v3',
                  'convnext_tiny', 'convnext_tiny_in22k', 'efficientnetv2_s',
                  'vgg16_bn', 'densenet121', 'mobilenet_v3_large', 'resnext50_32x4d'}


def run_specs():
    """Mọi run có summary.json + best.pt -> thông tin đủ để dựng lại model."""
    out = {}
    for name in sorted(os.listdir(RESULTS)):
        sp = os.path.join(RESULTS, name, 'summary.json')
        ck = os.path.join(RUNS, name, 'best.pt')
        if name.startswith('_') or not (os.path.exists(sp) and os.path.exists(ck)):
            continue
        s = json.load(open(sp, encoding='utf-8'))
        lora = int(name.rsplit('_lora', 1)[1]) if '_lora' in name else 0
        out[name] = {'model': s['model'], 'img_size': s['img_size'], 'lora': lora,
                     'top1': s['top1_accuracy'] * 100, 'ckpt': ck}
    return out


def build(spec):
    model, _ = models_zoo.build_model(spec['model'], pretrained=False,
                                      img_size=spec['img_size'])
    mean, std = model.norm_mean, model.norm_std
    if spec['lora']:
        model = models_zoo.apply_lora(model, spec['model'], spec['lora'])
    model.load_state_dict(torch.load(spec['ckpt'], map_location='cpu')['model'])
    return model.cuda().eval().to(memory_format=torch.channels_last), mean, std


def dataset(split, img_size, mean, std, train_img_size):
    image_dir = 'images_r512' if train_img_size >= 448 else 'images_r448'
    tf = build_transforms(img_size, train=False, mean=mean, std=std)
    ds = NABirds(split='train' if split == 'val' else 'test', transform=tf,
                 image_dir=image_dir)
    if split == 'val':
        # Cùng seed / cùng val_frac / cùng min_class => cùng tập val với lúc train.
        _, va = stratified_val_split(ds.samples, 0.15, seed=0, min_class_size=20)
        ds.samples = [ds.samples[k] for k in va]
    return ds


@torch.no_grad()
def logits_of(model, ds, batch_size, workers, flip):
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=workers, pin_memory=True)
    out, ys = [], []
    for x, y in loader:
        x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(x)
            o = o.logits if hasattr(o, 'logits') else o
            o = o.float()
            if flip:
                of = model(torch.flip(x, dims=[3]))
                of = of.logits if hasattr(of, 'logits') else of
                o = (o + of.float()) / 2
        out.append(o.cpu())
        ys.append(y)
    return torch.cat(out).numpy().astype(np.float16), torch.cat(ys).numpy()


def variants(spec):
    """Các biến thể TTA khả dụng cho một run: (tag, img_size, flip)."""
    r = spec['img_size']
    v = [('plain', r, False), ('flip', r, True)]
    if spec['model'] in VARIABLE_INPUT:
        big = int(round(r * 1.25 / 32) * 32)          # 224 -> 288, 299 -> 384
        # Eval transform là Resize(1.14 * size) + CenterCrop(size). Nếu cạnh đó
        # vượt cạnh ngắn của ảnh đã pre-resize thì TTA hoá ra đang đo trên ảnh
        # PHÓNG TO — thêm nhiễu nội suy chứ không thêm chi tiết. Với model @448
        # (cache 512px) thì res-TTA cần 657px, nên bỏ.
        cached_short = 512 if r >= 448 else 448
        if round(big * 1.14) <= cached_short:
            v += [('res', big, False), ('flipres', big, True)]
    return v


def cache(args):
    specs = run_specs()
    print(f'{len(specs)} run co checkpoint\n', flush=True)
    for name, spec in specs.items():
        todo = [(sp, t, r, f) for sp in ('val', 'test') for t, r, f in variants(spec)
                if not os.path.exists(os.path.join(RUNS, name, f'logits_{sp}_{t}.npy'))]
        if not todo:
            print(f'  {name}: da cache day du', flush=True)
            continue
        model, mean, std = build(spec)
        for split, tag, r, flip in todo:
            ds = dataset(split, r, mean, std, spec['img_size'])
            lg, ys = logits_of(model, ds, args.batch_size, args.workers, flip)
            np.save(os.path.join(RUNS, name, f'logits_{split}_{tag}.npy'), lg)
            ypath = os.path.join(RUNS, f'labels_{split}.npy')
            if not os.path.exists(ypath):
                np.save(ypath, ys)
            acc = (lg.astype(np.float32).argmax(1) == ys).mean() * 100
            # Chốt chặn: `plain` trên test PHẢI khớp con số đã ghi lúc train. Lệch
            # nghĩa là dựng lại model không khớp checkpoint — và loại lỗi này KHÔNG
            # ném exception (vd `transform_input` của inception/googlenet không nằm
            # trong state_dict nên `load_state_dict` vẫn thành công). Đã bắt được 2
            # lần theo cách này: inception mất 6.06 điểm, googlenet mất 7.11 điểm.
            if split == 'test' and tag == 'plain' and abs(acc - spec['top1']) > 0.5:
                raise ValueError(
                    f'{name}: dung lai model cho {acc:.2f}% nhung summary.json ghi '
                    f'{spec["top1"]:.2f}% (lech {acc - spec["top1"]:+.2f}). Kien truc '
                    f'dung lai KHONG khop checkpoint — sua truoc khi chay tiep.')
            print(f'  {name:28s} {split:4s} {tag:8s} r={r:3d} -> {acc:5.2f}', flush=True)
        del model
        torch.cuda.empty_cache()


def probs(name, split, tag):
    lg = np.load(os.path.join(RUNS, name, f'logits_{split}_{tag}.npy')).astype(np.float32)
    lg -= lg.max(1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(1, keepdims=True)


def ece(p, y, bins=15):
    """Expected Calibration Error: model tự tin tới mức nào so với mức nó đúng.

    Chia ảnh theo độ tin cậy (xác suất lớn nhất) thành `bins` khoảng, mỗi khoảng
    so accuracy thực với confidence trung bình, rồi lấy trung bình có trọng số.
    0 = hiệu chỉnh hoàn hảo. Ensemble thường hạ ECE vì trung bình xác suất làm
    mềm những dự đoán tự tin-mà-sai của từng model.
    """
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e * 100


def complementarity(pt, yt, rows, chosen, cur_t):
    """Vì sao ensemble ăn: hai model có sai CÙNG CHỖ không?

    `oracle` = tỉ lệ ảnh mà ÍT NHẤT MỘT trong hai model đúng — trần trên của mọi
    cách gộp hai model đó. Khoảng cách giữa oracle và accuracy của model tốt nhất
    chính là phần "model kia biết mà model này không biết". Nếu khoảng đó lớn thì
    ensemble còn dư địa; nếu nhỏ thì hai model sai giống hệt nhau và gộp vô ích.
    """
    best = max(rows, key=lambda r: r[1])[0]
    if best not in pt:
        return
    ok_best = pt[best].argmax(1) == yt
    print('\n### Vi sao ensemble an - do KHONG TRUNG LOI voi model tot nhat\n')
    print(f"{'model':30s} {'acc':>6s} {'ca hai sai':>11s} {'oracle':>7s} {'du dia':>7s}")
    ref = ok_best.mean() * 100
    out = []
    for n in pt:
        if n == best:
            continue
        ok = pt[n].argmax(1) == yt
        both_wrong = (~ok_best & ~ok).mean() * 100
        oracle = 100 - both_wrong
        out.append((oracle - ref, n, ok.mean() * 100, both_wrong, oracle))
    for gap, n, a, bw, orc in sorted(out, reverse=True)[:8]:
        print(f'{n:30s} {a:6.2f} {bw:11.2f} {orc:7.2f} {gap:+7.2f}')
    print(f'\n({best} mot minh: {ref:.2f}. "du dia" = oracle - {ref:.2f}, tuc phan '
          f'model kia dung ma no sai.)')

    print('\n### Hieu chinh xac suat (ECE, thap hon = tot hon)\n')
    print(f'  {best:32s} ECE {ece(pt[best], yt):5.2f}%')
    print(f'  {f"ensemble {len(chosen)} model":32s} ECE {ece(cur_t, yt):5.2f}%')


def report(args):
    specs = run_specs()
    yv = np.load(os.path.join(RUNS, 'labels_val.npy'))
    yt = np.load(os.path.join(RUNS, 'labels_test.npy'))

    def acc(p, y):
        return (p.argmax(1) == y).mean() * 100

    print('\n### TTA - chon bien the tot nhat tren VAL, bao cao tren TEST\n')
    print(f"{'run':30s} {'base-test':>9s} {'TTA chon':>9s} {'test-TTA':>9s} {'chenh':>7s}")
    best_tag, rows = {}, []
    for name, spec in specs.items():
        cand = [t for t, _, _ in variants(spec)
                if os.path.exists(os.path.join(RUNS, name, f'logits_val_{t}.npy'))]
        if not cand:
            continue
        va = {t: acc(probs(name, 'val', t), yv) for t in cand}
        pick = max(va, key=va.get)
        best_tag[name] = pick
        base = acc(probs(name, 'test', 'plain'), yt)
        tta = acc(probs(name, 'test', pick), yt)
        rows.append((name, base, pick, tta, tta - base))
        print(f'{name:30s} {base:9.2f} {pick:>9s} {tta:9.2f} {tta - base:+7.2f}')

    print('\n### Ensemble - chon thanh vien THAM LAM tren VAL, bao cao tren TEST\n')
    pool = list(best_tag)
    pv = {n: probs(n, 'val', best_tag[n]) for n in pool}
    pt = {n: probs(n, 'test', best_tag[n]) for n in pool}
    # Tập val chỉ 3,510 ảnh. Nếu cứ thấy val nhích lên là thêm model thì sẽ đuổi
    # theo nhiễu: lần chạy đầu greedy thêm tới 4 model vì val tăng
    # 93.33 -> 93.45 -> 93.59 -> 93.62, nhưng TEST lại đi xuống sau bước 2
    # (90.00 -> 90.96 -> 90.76 -> 90.48). Hai bước cuối chỉ là nhiễu val.
    # Nên: chỉ thêm khi mức tăng vượt SAI SỐ CHUẨN GHÉP CẶP (McNemar) của chính
    # phép so đó — se = sqrt(số ảnh hai bên bất đồng) / n.
    # Quỹ đạo được in ĐẦY ĐỦ tới 6 model, nhưng điểm dừng do VAL quyết định và
    # được đánh dấu rõ. In tiếp phần sau điểm dừng là để người đọc thấy val có
    # phân giải được các khác biệt này hay không — KHÔNG phải để chọn theo test.
    # Lần chạy thực tế: val chỉ tăng +0.11 khi thêm model thứ hai (dưới sai số
    # +-0.30) trong khi TEST tăng +0.96. Kết luận trung thực là **tập val 3,510
    # ảnh quá nhỏ để xếp hạng ensemble**, chứ không phải "ensemble vô dụng".
    chosen, cur_v, best_v, stop_at = [], None, -1.0, None
    while True:
        cands = [n for n in pool if n not in chosen]
        if not cands:
            break
        scored = []
        for n in cands:
            mix = pv[n] if not chosen else (cur_v * len(chosen) + pv[n]) / (len(chosen) + 1)
            scored.append((acc(mix, yv), n))
        a, n = max(scored)
        if chosen:
            new_mix = (cur_v * len(chosen) + pv[n]) / (len(chosen) + 1)
            ok_old, ok_new = cur_v.argmax(1) == yv, new_mix.argmax(1) == yv
            disc = int((ok_old != ok_new).sum())
            se = (disc ** 0.5) / len(yv) * 100
            if a - best_v <= se and stop_at is None:
                stop_at = len(chosen)
                print(f'  [QUY TAC DUNG O DAY] them {n} chi duoc {a - best_v:+.2f} val, '
                      f'khong vuot sai so ghep cap +-{se:.2f}. Cac dong duoi CHI de '
                      f'xem, KHONG duoc chon bang test.')
        elif a <= best_v:
            break
        if len(chosen) >= 6:
            break
        best_v = a
        cur_v = pv[n] if not chosen else (cur_v * len(chosen) + pv[n]) / (len(chosen) + 1)
        chosen.append(n)
        cur_t = sum(pt[c] for c in chosen) / len(chosen)
        print(f'  +{n:30s} val {a:6.2f}  |  test {acc(cur_t, yt):6.2f}  '
              f'({len(chosen)} model)', flush=True)

    n_sel = stop_at if stop_at is not None else len(chosen)
    chosen = chosen[:n_sel]
    cur_t = sum(pt[c] for c in chosen) / len(chosen)
    solo = max(rows, key=lambda r: r[1])
    ens = acc(cur_t, yt)
    complementarity(pt, yt, rows, chosen, cur_t)
    print(f'\nEnsemble {len(chosen)} model -> test top-1 {ens:.2f} '
          f'(model don tot nhat: {solo[0]} {solo[1]:.2f}, chenh {ens - solo[1]:+.2f})')

    ds = dataset('test', 224, models_zoo.IMAGENET_MEAN, models_zoo.IMAGENET_STD, 224)
    pred = cur_t.argmax(1)
    top5 = np.argsort(-cur_t, axis=1)[:, :5]
    compute_all(yt, pred, (top5 == yt[:, None]).any(1), class_names=ds.idx_to_name,
                species=ds.idx_to_species, orders=ds.idx_to_order,
                out_dir=os.path.join(RESULTS, 'ensemble'), run_name='ensemble',
                extra={'model': 'ensemble', 'img_size': 224, 'params_M': 0.0,
                       'pretrained': True, 'use_bbox': False, 'epochs': 0,
                       'batch_size': 0, 'train_minutes': 0.0,
                       'members': chosen, 'tta': {c: best_tag[c] for c in chosen}})
    print('-> results/ensemble/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    if not (a.cache or a.report):
        ap.error('can --cache hoac --report')
    if a.cache:
        cache(a)
    if a.report:
        report(a)
