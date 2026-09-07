"""Train / đánh giá NABirds 555 lớp.

Ví dụ:
  python src/train.py --model resnet50 --epochs 15
  python src/train.py --model inception_v3 --epochs 15 --batch-size 48
  python src/train.py --model cnn_scratch --epochs 30 --lr-head 0.05
  python src/train.py --model resnet50 --eval-only --ckpt runs/resnet50/best.pt
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

import models_zoo
from metrics import compute_all, format_summary
from nabirds_data import build_loaders

REPO = models_zoo.REPO


def parse_args():
    p = argparse.ArgumentParser(description='NABirds 555-way classification')
    p.add_argument('--model', default='resnet50', choices=models_zoo.MODEL_NAMES)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--img-size', type=int, default=None, help='mặc định theo model')
    p.add_argument('--lr-backbone', type=float, default=None)
    p.add_argument('--lr-head', type=float, default=None)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--optimizer', default='sgd', choices=['sgd', 'adamw'])
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--freeze-epochs', type=int, default=1,
                   help='số epoch đầu đóng băng backbone, chỉ train head')
    p.add_argument('--lora-rank', type=int, default=0,
                   help='>0: đóng băng backbone, chỉ train adapter LoRA + head. '
                        'Chỉ hỗ trợ vit_b_16_in21k (timm) vì ở đó q/k/v là MỘT '
                        'nn.Linear tên `qkv`; torchvision/open_clip gộp q/k/v vào '
                        'nn.MultiheadAttention nên peft không gắn vào được.')
    p.add_argument('--grad-checkpointing', action='store_true',
                   help='đánh đổi ~30%% thời gian lấy VRAM. Cần cho ViT-B/16 @448 '
                        'trên 12GB: 448/16 = 28x28 = 784 token thay vì 196.')
    p.add_argument('--warmup-steps', type=int, default=300)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--use-bbox', action='store_true',
                   help='thí nghiệm đối chứng: crop theo bbox (KHÔNG phải setting chuẩn)')
    p.add_argument('--aug', default='standard', choices=['standard', 'light'])
    # --- các núm cho thí nghiệm augmentation (PLAN mục E) ---
    # Mặc định giữ NGUYÊN hành vi cũ để mọi run đã chạy vẫn so được.
    p.add_argument('--erasing-p', type=float, default=0.25,
                   help='E1: xác suất RandomErasing. 0 = tắt. Tài liệu ghi Cutout '
                        'làm ResNet-50 mất ~2 điểm trên CUB vì xoá trúng vùng phân biệt.')
    p.add_argument('--hue', type=float, default=0.02,
                   help='E4: biên độ hue jitter. Để 0.02 vì D3 — MÀU BỘ LÔNG CHÍNH LÀ '
                        'NHÃN (Yellow vs Orange Warbler). Chỉ tăng khi chạy đối chứng.')
    p.add_argument('--rotate', type=float, default=0.0,
                   help='E2: biên độ xoay (độ). 0 = tắt. Chim có hướng chuẩn nên <= 15.')
    p.add_argument('--mixup-alpha', type=float, default=0.0,
                   help='E3: alpha của Beta cho mixup. 0 = tắt.')
    p.add_argument('--cutmix-alpha', type=float, default=0.0,
                   help='E3: alpha của Beta cho cutmix. 0 = tắt.')
    p.add_argument('--cmo-p', type=float, default=0.0,
                   help='E2/CMO: xác suất dán chim LỚP HIẾM (bbox thật) lên ảnh lớp '
                        'nhiều ảnh. Park et al. CVPR 2022 — lớp thiểu số thiếu BỐI '
                        'CẢNH đa dạng nên mượn nền của lớp đa số. 0 = tắt.')
    p.add_argument('--cmo-tail-max', type=int, default=30,
                   help='lớp có ít hơn N ảnh train được coi là lớp hiếm (113/555 lớp).')
    p.add_argument('--mix-prob', type=float, default=0.5,
                   help='E3: xác suất áp dụng trộn ở mỗi batch (khi có alpha > 0). '
                        'CHỈ có nghĩa khi train DÀI: benchmark OpenMixup trên CUB đo ở '
                        '200 epoch; ở 25 epoch mixup chưa kịp phát huy.')
    p.add_argument('--run-name', default=None)
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--ckpt', default=None)
    p.add_argument('--image-dir', default='images_r448',
                   help='thư mục ảnh đã pre-resize dưới nabirds/ (vd images_r512 cho train 448)')
    p.add_argument('--val-min-class', type=int, default=20,
                   help='lớp có ít hơn N ảnh train sẽ KHÔNG góp ảnh cho val — '
                        'ưu tiên dữ liệu cho train (39/555 lớp, giữ 100%% ảnh)')
    p.add_argument('--val-frac', type=float, default=0.15,
                   help='phần tập train tách ra làm validation (0 = chọn checkpoint '
                        'bằng chính test, nhanh hơn nhưng con số bị lạc quan)')
    p.add_argument('--patience', type=int, default=8,
                   help='dừng sớm nếu test top-1 không cải thiện sau N epoch (0 = tắt)')
    p.add_argument('--min-delta', type=float, default=0.0,
                   help='muc cai thien toi thieu de reset bo dem patience. '
                        'PHAI de 0: dat > 0 se lam bo dem khong bao gio reset khi '
                        'model tien bo tung buoc nho (resnet50 tung bi dung o epoch '
                        '17/30 du val van tang deu 81.5%% -> 83.0%%). Chong nhieu la '
                        'viec cua --patience, khong phai cua nguong nay.')
    p.add_argument('--max-steps', type=int, default=0,
                   help='gioi han so step/epoch (0 = khong gioi han). Dung de smoke test.')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


DEFAULT_LR = {  # (lr_backbone, lr_head) cho SGD momentum 0.9
    'cnn_scratch':    (0.05, 0.05),
    'resnet50':       (0.005, 0.05),
    'resnet101':      (0.005, 0.05),
    'inception_v3':   (0.005, 0.05),
    'vit_b_16':       (0.001, 0.01),
    'vit_b_16_in21k': (0.001, 0.01),
    # CNN hiện đại: bảng này CHỈ là mức an toàn khi chạy tay bằng SGD, KHÔNG phải
    # recipe được khuyến nghị. Probe 4 epoch (runs/logs/_probe_cnx.log) đo được:
    #   convnext_tiny + SGD   (1e-3/1e-2)  -> val 32.54
    #   convnext_tiny + AdamW (1e-4/1e-3)  -> val 81.91   <-- chênh 49 ĐIỂM
    # Cùng mức lr SGD đó lại chạy tốt cho vit_b_16_in21k (val 86.84 @ep4), nên đây
    # không phải chuyện LayerNorm — chỉ là mức lr không chuyển từ ViT sang ConvNeXt
    # được. Dùng `src/run_cnn2.sh` (truyền --optimizer adamw tường minh).
    'convnext_tiny':       (0.001, 0.01),
    'convnext_tiny_in22k': (0.001, 0.01),
    'efficientnetv2_s':    (0.001, 0.01),
    'swin_tiny_in22k':     (0.001, 0.01),
    # BioCLIP: CÙNG lr với vit_b_16_in21k, cố ý. Cùng kiến trúc ViT-B/16, cùng
    # optimizer, cùng lr -> chênh lệch quy được về NGUỒN PRETRAIN, không lẫn gì khác.
    'bioclip':            (0.001, 0.01),
    # CNN BatchNorm cổ điển: cùng mức lr với resnet50 để nằm chung một trục.
    'vgg16_bn':           (0.002, 0.02),      # VGG không có skip connection -> lr thấp hơn
    'densenet121':        (0.005, 0.05),
    'mobilenet_v3_large': (0.005, 0.05),
    'resnext50_32x4d':    (0.005, 0.05),
    'googlenet':          (0.005, 0.05),
    'efficientnet_b0':    (0.005, 0.05),
    'alexnet':            (0.002, 0.02),      # không có BatchNorm, như vgg16_bn
    'vit_small_in21k':    (0.001, 0.01),      # giống hệt vit_b_16_in21k, cố ý
}


def mix_batch(x, y, mixup_alpha, cutmix_alpha, mix_prob):
    """Mixup / CutMix cho một batch. Trả về (x, y_a, y_b, lam).

    lam = 1.0 nghĩa là KHÔNG trộn, khi đó y_b bị bỏ qua ở phía loss.

    Vì sao nhãn dùng tỉ lệ DIỆN TÍCH cho cutmix: đó là công thức gốc. Với FGVC nó
    là điểm yếu đã biết — vùng phân biệt của chim rất nhỏ nên cắt một mảng lớn
    nền rồi gán 50% nhãn là nhiễu nhãn. SnapMix (AAAI'21) sửa đúng chỗ này bằng
    CAM. Ở đây giữ bản gốc trước để có mốc so sánh sạch.
    """
    import random
    if mix_prob <= 0 or random.random() > mix_prob:
        return x, y, y, 1.0
    use_cut = cutmix_alpha > 0 and (mixup_alpha <= 0 or random.random() < 0.5)
    alpha = cutmix_alpha if use_cut else mixup_alpha
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if use_cut:
        h, w = x.shape[2:]
        rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
        x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - ((y2 - y1) * (x2 - x1) / (h * w))     # lam THẬT theo diện tích
    else:
        x = lam * x + (1 - lam) * x[perm]
    return x, y, y[perm], lam


def make_scheduler(optimizer, total_steps, warmup_steps):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate(model, loader, device, collect=False):
    model.eval()
    n = correct1 = correct5 = 0
    ys, ps, t5 = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
        if isinstance(out, (tuple, list)) or hasattr(out, 'logits'):
            out = out[0] if isinstance(out, (tuple, list)) else out.logits
        out = out.float()
        top5 = out.topk(5, dim=1).indices
        pred = top5[:, 0]
        hit5 = (top5 == y[:, None]).any(1)
        correct1 += (pred == y).sum().item()
        correct5 += hit5.sum().item()
        n += y.numel()
        if collect:
            ys.append(y.cpu()); ps.append(pred.cpu()); t5.append(hit5.cpu())
    acc1, acc5 = correct1 / n, correct5 / n
    if not collect:
        return acc1, acc5, None
    return acc1, acc5, (torch.cat(ys).numpy(), torch.cat(ps).numpy(),
                        torch.cat(t5).numpy())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, default_size = models_zoo.build_model(
        args.model, pretrained=not args.no_pretrained, img_size=args.img_size)
    norm_mean, norm_std = model.norm_mean, model.norm_std   # lấy TRƯỚC khi bọc LoRA
    if args.lora_rank:
        model = models_zoo.apply_lora(model, args.model, args.lora_rank)
    if args.grad_checkpointing:
        if not hasattr(model, 'set_grad_checkpointing'):
            raise ValueError(f'{args.model} khong ho tro grad checkpointing')
        model.set_grad_checkpointing()
    img_size = args.img_size or default_size
    run_name = args.run_name or (f'{args.model}_{img_size}'
                                 + (f'_lora{args.lora_rank}' if args.lora_rank else '')
                                 + ('_bbox' if args.use_bbox else ''))
    run_dir = os.path.join(REPO, 'runs', run_name)
    res_dir = os.path.join(REPO, 'results', run_name)
    os.makedirs(run_dir, exist_ok=True)

    model = model.to(device).to(memory_format=torch.channels_last)
    n_params = models_zoo.count_parameters(model)
    # build_model ép cnn_scratch về không-pretrain, nên cờ hiển thị phải theo
    # trạng thái thật chứ không theo tham số dòng lệnh.
    pretrained = (not args.no_pretrained) and args.model != 'cnn_scratch'

    train_set, val_set, test_set, train_loader, val_loader, test_loader = build_loaders(
        img_size, args.batch_size, args.workers, args.use_bbox, args.aug,
        val_frac=args.val_frac, seed=args.seed, image_dir=args.image_dir,
        val_min_class=args.val_min_class,
        mean=norm_mean, std=norm_std,
        erasing_p=args.erasing_p, hue=args.hue, rotate=args.rotate,
        cmo_p=args.cmo_p, cmo_tail_max=args.cmo_tail_max)
    # Tập dùng để chọn checkpoint / early stopping.
    sel_loader = val_loader if val_loader is not None else test_loader
    sel_name = 'val' if val_loader is not None else 'test'

    print(f'== {run_name} ==', flush=True)
    print(f'  model {args.model} | params {n_params/1e6:.1f}M | img {img_size} '
          f'| pretrained {pretrained} | bbox {args.use_bbox}', flush=True)
    print(f'  train {len(train_set)} | val {len(val_set) if val_set else 0} '
          f'| test {len(test_set)} | bs {args.batch_size} | epochs {args.epochs} '
          f'| patience {args.patience} | chọn checkpoint theo {sel_name.upper()}', flush=True)

    if args.eval_only:
        ckpt = args.ckpt or os.path.join(run_dir, 'best.pt')
        model.load_state_dict(torch.load(ckpt, map_location=device)['model'])
        acc1, acc5, packed = evaluate(model, test_loader, device, collect=True)
        finalize(packed, test_set, res_dir, run_name, args, n_params, img_size, 0.0,
                 pretrained=pretrained)
        return

    lr_b, lr_h = DEFAULT_LR[args.model]
    lr_b = args.lr_backbone if args.lr_backbone is not None else lr_b
    lr_h = args.lr_head if args.lr_head is not None else lr_h
    groups = models_zoo.param_groups(model, args.model, lr_b, lr_h, args.weight_decay)
    if args.optimizer == 'sgd':
        opt = torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    else:
        opt = torch.optim.AdamW(groups)
    steps_per_epoch = len(train_loader)
    sched = make_scheduler(opt, args.epochs * steps_per_epoch, args.warmup_steps)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    do_mix = args.mixup_alpha > 0 or args.cutmix_alpha > 0
    if do_mix and args.epochs < 40:
        print(f'  CANH BAO: bat mixup/cutmix voi chi {args.epochs} epoch. Benchmark '
              f'tren CUB do o 200 epoch; duoi ~40 epoch mixup thuong CHUA kip co lai, '
              f'nen ket qua am o day KHONG chung minh duoc mixup vo dung.', flush=True)

    def mix_loss(crit, logits, y_a, y_b, lam):
        if lam >= 1.0:
            return crit(logits, y_a)
        return lam * crit(logits, y_a) + (1 - lam) * crit(logits, y_b)

    def cmo_loss(crit, logits, y, y_paste, lam):
        """lam theo TỪNG ẢNH (không phải theo batch như mixup) vì diện tích dán
        khác nhau ở mỗi ảnh — nên phải dùng reduction='none' rồi mới trộn."""
        if y_paste is None:
            return None
        import torch.nn.functional as _F
        ls = crit.label_smoothing
        a = _F.cross_entropy(logits, y, label_smoothing=ls, reduction='none')
        b = _F.cross_entropy(logits, y_paste, label_smoothing=ls, reduction='none')
        return (lam * a + (1 - lam) * b).mean()

    head_keys = models_zoo.head_parameter_names(model, args.model)

    def set_backbone_trainable(flag):
        for pname, p in model.named_parameters():
            is_head = any(pname.startswith(h) or f'.{h}' in pname for h in head_keys)
            p.requires_grad_(True if is_head else flag)

    best = 0.0
    best_epoch = 0
    since_improve = 0
    stopped_early = False
    history = []
    t_start = time.time()
    for epoch in range(args.epochs):
        # Với LoRA, backbone đã đóng băng vĩnh viễn và chỉ adapter + head train được,
        # nên KHÔNG gọi set_backbone_trainable (nó sẽ mở khoá lại toàn bộ trọng số gốc).
        if (args.freeze_epochs and not args.no_pretrained
                and args.model != 'cnn_scratch' and not args.lora_rank):
            set_backbone_trainable(epoch >= args.freeze_epochs)
        model.train()
        # Cộng dồn loss/số dự đoán đúng trên GPU. Gọi .item() mỗi step sẽ ép
        # đồng bộ CPU-GPU và chặn overlap giữa data loading và compute, nên chỉ
        # .item() ở mốc in log (6 lần/epoch) và ở cuối epoch.
        run_loss = torch.zeros((), device=device)
        run_correct = torch.zeros((), device=device, dtype=torch.long)
        seen = 0
        t0 = time.time()
        n_steps = min(steps_per_epoch, args.max_steps) if args.max_steps else steps_per_epoch
        log_every = max(1, n_steps // 3) if args.max_steps else max(20, steps_per_epoch // 6)
        for step, batch in enumerate(train_loader):
            if args.cmo_p > 0:               # dataset trả 4 phần tử khi bật CMO
                x, y, y_paste, lam_cmo = batch
                y_paste = y_paste.to(device, non_blocking=True)
                lam_cmo = lam_cmo.to(device, non_blocking=True).float()
            else:
                x, y = batch
                y_paste, lam_cmo = None, None
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            x, y_a, y_b, lam = mix_batch(x, y, args.mixup_alpha, args.cutmix_alpha,
                                         args.mix_prob if do_mix else 0.0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(x)
                if model.is_inception and model.training:
                    logits = out.logits
                    loss = (mix_loss(crit, logits, y_a, y_b, lam)
                            + 0.4 * mix_loss(crit, out.aux_logits, y_a, y_b, lam))
                else:
                    logits = out.logits if hasattr(out, 'logits') else out
                    loss = (cmo_loss(crit, logits, y, y_paste, lam_cmo)
                            if args.cmo_p > 0 else
                            mix_loss(crit, logits, y_a, y_b, lam))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run_loss += loss.detach() * y.numel()
            # train-acc luôn tính theo nhãn GỐC: nếu tính theo nhãn đã trộn thì
            # con số nhảy lung tung và không so được với các run không trộn.
            run_correct += (logits.detach().argmax(1) == y).sum()
            seen += y.numel()
            if args.max_steps and step + 1 >= args.max_steps:
                break_after = True
            else:
                break_after = False
            if (step + 1) % log_every == 0 or break_after:
                el = time.time() - t0
                print(f'    [{epoch+1}] step {step+1:4d}/{steps_per_epoch}  '
                      f'loss(avg) {run_loss.item()/seen:.3f}  '
                      f'loss(batch) {loss.item():.3f}  '
                      f'train-acc {run_correct.item()/seen*100:5.2f}%  '
                      f'lr {opt.param_groups[-1]["lr"]:.2e}  '
                      f'{seen/el:.0f} img/s  eta {(n_steps-step-1)*el/(step+1):.0f}s',
                      flush=True)
            if break_after:
                break
        tr_time = time.time() - t0
        run_loss = run_loss.item()
        tr_acc = run_correct.item() / seen
        t_ev = time.time()
        acc1, acc5, _ = evaluate(model, sel_loader, device)
        ev_time = time.time() - t_ev
        history.append({'epoch': epoch + 1, 'train_loss': run_loss / seen,
                        'train_acc': tr_acc, 'top1': acc1, 'top5': acc5,
                        'train_sec': tr_time, 'eval_sec': ev_time})
        star = ''
        if acc1 > best + args.min_delta:
            since_improve = 0
        else:
            since_improve += 1
        if acc1 > best:
            best = acc1
            best_epoch = epoch + 1
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'top1': acc1, 'args': vars(args)},
                       os.path.join(run_dir, 'best.pt'))
            star = ' *'
        print(f'  ep {epoch+1:02d}/{args.epochs}  train-loss {run_loss/seen:.3f}  '
              f'train-acc {tr_acc*100:5.2f}  |  {sel_name}-top1 {acc1*100:5.2f}  '
              f'{sel_name}-top5 {acc5*100:5.2f}  |  train {tr_time:.0f}s '
              f'+ eval {ev_time:.0f}s{star}'
              f'{"" if (since_improve == 0 or not args.patience) else f"  (no-improve {since_improve}/{args.patience})"}',
              flush=True)
        if args.patience and since_improve >= args.patience:
            print(f'  early stopping: {sel_name} top-1 không cải thiện {since_improve} epoch '
                  f'liên tiếp (best {best*100:.2f}% @ ep {best_epoch})', flush=True)
            stopped_early = True
            break

    total_min = (time.time() - t_start) / 60
    ckpt_path = os.path.join(run_dir, 'best.pt')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
    else:                      # khong epoch nao cai thien -> dung trong so cuoi cung
        print('  CANH BAO: khong co best.pt, danh gia bang trong so epoch cuoi', flush=True)
    _, _, packed = evaluate(model, test_loader, device, collect=True)
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    finalize(packed, test_set, res_dir, run_name, args, n_params, img_size, total_min,
             pretrained=pretrained,
             extra={'epochs_run': len(history), 'best_epoch': best_epoch,
                    'stopped_early': stopped_early, 'selection_set': sel_name,
                    'val_frac': args.val_frac, 'n_train': len(train_set),
                    f'best_{sel_name}_top1': best})


def finalize(packed, test_set, res_dir, run_name, args, n_params, img_size, minutes,
             extra=None, pretrained=None):
    y_true, y_pred, t5 = packed
    summary, per_class = compute_all(
        y_true, y_pred, t5,
        class_names=test_set.idx_to_name,
        species=test_set.idx_to_species,
        orders=test_set.idx_to_order,
        out_dir=res_dir, run_name=run_name,
        extra={'model': args.model, 'img_size': img_size,
               'params_M': round(n_params / 1e6, 2),
               'pretrained': bool(pretrained),
               'use_bbox': args.use_bbox, 'epochs': args.epochs,
               'batch_size': args.batch_size, 'train_minutes': round(minutes, 1),
               **(extra or {})})
    print(format_summary(summary), flush=True)
    print(f'  -> {res_dir}', flush=True)


if __name__ == '__main__':
    main()
