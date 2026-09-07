"""Bước 2b — đánh giá zero-shot: embedding text x embedding ảnh đã cache.

Giao thức bám `examples/zero_shot.py` của BioCLIP:
  1. mỗi lớp -> 80 template OpenAI -> encode_text -> L2-norm -> TRUNG BÌNH -> L2-norm
  2. logits = image_features @ text_features.T   (cả hai đã L2-norm)
  3. argmax

Hai không gian nhãn, báo cáo song song:
  `leaf`    555 lớp lá. Đây là bài gốc của NABirds. Prompt chỉ mang tên loài bị
            chặn trần 78.75% (README 1.2) -> con số này đo đúng khoảng cách đó.
  `species` 404 loài. Không gian mà taxonomy đủ sức phân biệt.

Với `leaf`, tái dụng `src/metrics.py` của bài có giám sát nên mọi cột (bal-acc,
macro-F1, acc@404sp, acc@22ord) so trực tiếp được với bảng kết quả 8 run đã có.
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
import zs_models  # noqa: E402
import prompts  # noqa: E402
import encode as enc  # noqa: E402
import nabirds_io as nio  # noqa: E402
from metrics import compute_all  # noqa: E402


def text_features(model, tokenizer, texts, device, batch=256):
    """80 template mỗi lớp -> 1 vector/lớp. Trả về (n_class, D) đã L2-norm."""
    flat, k = prompts.expand(texts)
    outs = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(flat), batch):
            tok = tokenizer(flat[i:i + batch]).to(device)
            outs.append(torch.nn.functional.normalize(
                model.encode_text(tok), dim=-1).float())
    f = torch.cat(outs).reshape(len(texts), k, -1).mean(1)
    return torch.nn.functional.normalize(f, dim=-1)


def leaf_groupings():
    """(tên 555 lá, species của mỗi lá, order tiếng Anh của mỗi lá) theo thứ tự nhãn."""
    class_names = nio.load_class_names()
    hierarchy = nio.load_hierarchy()
    labels = nio.load_image_labels()
    _, leaf_ids = nio.build_label_index(labels)
    tax = nio.build_taxonomy(leaf_ids, hierarchy, class_names)
    return ([class_names[c] for c in leaf_ids],
            [tax[c][0] for c in leaf_ids],
            [tax[c][1] for c in leaf_ids])


def leaf_to_species_index(df):
    """Nhãn lá 0..554 -> nhãn loài 0..403 (thứ tự như `class_texts(space='species')`)."""
    sp_ids = sorted(df.species_class_id.unique(), key=int)
    pos = {s: i for i, s in enumerate(sp_ids)}
    sub = df.sort_values('class_id', key=lambda s: s.astype(int))
    return np.array([pos[s] for s in sub.species_class_id], dtype=np.int64)


def oracle_variant_acc(logits, y_leaf, l2s, restrict=None, tag=''):
    """Phép đo TRUNG TÂM của dự án: cho sẵn đúng LOÀI, có chọn đúng BIẾN THỂ không?

    Với mỗi ảnh, thu hẹp không gian nhãn về đúng các lá của loài nó thuộc về, rồi
    argmax. Loại bỏ hoàn toàn nhiễu từ việc đoán loài — thứ mà BioCLIP đã thấy khi
    pretrain nên gần như luôn đúng.

    Chỉ tính trên ảnh của 137 loài đa-lá (11,955/24,633). Hai mốc để so:
      random  — đoán ngẫu nhiên trong các lá của loài đó
      prior   — luôn chọn lá đông nhất của loài (không dùng thông tin ảnh)
    """
    import collections
    leaves_of = collections.defaultdict(list)
    for leaf, sp in enumerate(l2s):
        leaves_of[sp].append(leaf)
    full = np.array([len(leaves_of[l2s[t]]) > 1 for t in y_leaf])
    m = full & restrict[y_leaf] if restrict is not None else full
    if m.sum() == 0:
        return {}

    # Mốc prior phải đếm trên TOÀN BỘ tập đa-lá, không phải trên tập đã lọc: nếu
    # đếm sau khi lọc về riêng các lá unseen thì lá unseen luôn là lá đông nhất
    # trong chính nó -> prior = 100%, vô nghĩa.
    cnt = collections.Counter(y_leaf[full].tolist())
    hit, rnd, pri = [], [], []
    for i in np.where(m)[0]:
        cand = leaves_of[l2s[y_leaf[i]]]
        hit.append(cand[int(np.argmax(logits[i, cand]))] == y_leaf[i])
        rnd.append(1.0 / len(cand))
        pri.append(max(cand, key=lambda c: cnt.get(c, 0)) == y_leaf[i])
    return {f'oracle_variant{tag}_n': int(m.sum()),
            f'oracle_variant{tag}_acc': float(np.mean(hit)),
            f'oracle_variant{tag}_random': float(np.mean(rnd)),
            f'oracle_variant{tag}_prior': float(np.mean(pri)),
            f'oracle_variant{tag}_gain': float(np.mean(hit)) - float(np.mean(pri))}


def variant_probe(logits, y_leaf, l2s, species_subset=None, tag=''):
    """Phép đo TRUNG TÂM: cho sẵn đúng LOÀI, có chọn đúng BIẾN THỂ không?

    Thu hẹp không gian nhãn về đúng các lá của loài mà ảnh thuộc về, rồi argmax.
    Bỏ hẳn nhiễu từ việc đoán loài — thứ mà BioCLIP đã thấy khi pretrain.

    BÁO CÁO CHỈ SỐ CÂN BẰNG THEO LÁ (`_bal`), không phải theo ảnh. Lý do: nếu chỉ
    tính trên các lá female (81 lá unseen) thì một model thiên vị "female" ăn điểm
    cao giả — đo thử cách đó cho CLIP T3a 85.68 trong khi trên toàn bộ 137 loài đa-lá
    nó chỉ được 54.89. Chỉ số cân bằng lấy trung bình accuracy CỦA TỪNG LÁ nên
    thiên vị một biến thể sẽ tự triệt tiêu. Mốc ngẫu nhiên = trung bình 1/số-lá.
    """
    import collections
    leaves_of = collections.defaultdict(list)
    for leaf, sp in enumerate(l2s):
        leaves_of[sp].append(leaf)

    per_leaf, chance = {}, {}
    for leaf in range(len(l2s)):
        cand = leaves_of[l2s[leaf]]
        if len(cand) < 2:
            continue
        if species_subset is not None and l2s[leaf] not in species_subset:
            continue
        m = y_leaf == leaf
        if not m.any():
            continue
        per_leaf[leaf] = float(np.mean(
            np.asarray(cand)[logits[np.ix_(m, cand)].argmax(1)] == leaf))
        chance[leaf] = 1.0 / len(cand)
    if not per_leaf:
        return {}
    return {f'vp{tag}_n_leaves': len(per_leaf),
            f'vp{tag}_bal': float(np.mean(list(per_leaf.values()))),
            f'vp{tag}_chance': float(np.mean(list(chance.values())))}


def variant_metrics(y_leaf, y_pred_leaf, l2s):
    """Đo riêng phần BIẾN THỂ — câu hỏi trung tâm của dự án.

    Chỉ xét ảnh thuộc 137 species có nhiều hơn 1 lá (11,955/24,633 ảnh test).
    Với các ảnh đó, khi model đã đoán ĐÚNG loài, nó có chọn đúng biến thể không?

    `variant_prior` là mốc phải vượt: chiến lược "luôn chọn lá đông nhất của loài",
    không dùng thông tin ảnh nào cả. Không vượt được mốc này nghĩa là text không
    mang thêm tín hiệu thị giác nào về giới tính/tuổi/bộ lông.
    """
    import collections
    n_leaf_per_sp = collections.Counter(l2s.tolist())
    multi = np.array([n_leaf_per_sp[s] > 1 for s in l2s])   # (555,) lá thuộc species đa-lá
    m = multi[y_leaf]                                        # (N,) ảnh thuộc species đa-lá
    if m.sum() == 0:
        return {}

    sp_ok = l2s[y_pred_leaf] == l2s[y_leaf]
    sel = m & sp_ok
    cnt = collections.Counter(y_leaf[m].tolist())
    top_leaf = {}
    for leaf, c in cnt.items():
        s = l2s[leaf]
        if c > cnt.get(top_leaf.get(s, leaf), -1) or s not in top_leaf:
            top_leaf[s] = leaf
    prior_hit = np.array([top_leaf[l2s[t]] == t for t in y_leaf[m]])

    return {
        'n_multileaf_imgs': int(m.sum()),
        'species_acc_on_multileaf': float(sp_ok[m].mean()),
        'variant_acc': float((y_pred_leaf[sel] == y_leaf[sel]).mean()) if sel.sum() else 0.0,
        'variant_prior': float(prior_hit.mean()),
        'variant_gain': (float((y_pred_leaf[sel] == y_leaf[sel]).mean()) - float(prior_hit.mean())
                         if sel.sum() else 0.0),
    }


def evaluate(img_feats, y_true, txt_feats, top_k=5):
    """-> (y_pred, top5_correct, logits_top5). Chạy theo lô để không nổ VRAM."""
    x = torch.from_numpy(img_feats).cuda()
    w = txt_feats.cuda().T
    preds, top5, allz = [], [], []
    for i in range(0, len(x), 4096):
        logits = x[i:i + 4096] @ w
        k = min(top_k, logits.shape[1])
        idx = logits.topk(k, dim=1).indices
        preds.append(idx[:, 0].cpu())
        yt = torch.from_numpy(y_true[i:i + 4096]).cuda()
        top5.append((idx == yt[:, None]).any(1).cpu())
        allz.append(logits.cpu())
    return (torch.cat(preds).numpy(), torch.cat(top5).numpy(),
            torch.cat(allz).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['clip_b16', 'bioclip', 'bioclip2'])
    ap.add_argument('--levels', nargs='+', default=['T0', 'T1', 'T2'])
    ap.add_argument('--taxo', default='auto',
                    help="'auto' = ToL cho bioclip*, GBIF cho clip; hoac ep 'tol'/'gbif'")
    ap.add_argument('--split', default='test')
    args = ap.parse_args()

    df = prompts.load_tables()
    names555, species555, orders555 = leaf_groupings()
    l2s = leaf_to_species_index(df)
    import splits as sp_mod                     # 81 loài của split `variant`
    unseen_mask = sp_mod.variant_split(df)
    sub81 = set(l2s[np.where(unseen_mask)[0]].tolist())
    device = 'cuda'
    rows = []

    for mname in args.models:
        emb_path, lab_path = enc.paths(mname, args.split)
        if not os.path.exists(emb_path):
            print(f'!! thieu {emb_path}, bo qua {mname}')
            continue
        img = np.load(emb_path)
        y_leaf = np.load(lab_path)
        y_sp = l2s[y_leaf]
        model, _, tokenizer = zs_models.load(mname, device)
        taxo = args.taxo if args.taxo != 'auto' else ('tol' if 'bioclip' in mname else 'gbif')

        for level in args.levels:
            for space, y in (('leaf', y_leaf), ('species', y_sp)):
                texts, _ = prompts.class_texts(df, level, space=space, taxo=taxo)
                tf = text_features(model, tokenizer, texts, device)
                y_pred, top5, logits = evaluate(img, y, tf)
                run = f'{mname}_{level}_{space}'

                if space == 'leaf':
                    extra = {'model': mname, 'level': level, 'space': space, 'taxo': taxo,
                             'n_unique_texts': len(set(texts))}
                    extra.update(variant_metrics(y, y_pred, l2s))
                    extra.update(oracle_variant_acc(logits, y, l2s))
                    extra.update(variant_probe(logits, y, l2s))
                    extra.update(variant_probe(logits, y, l2s, sub81, '81'))
                    s, _ = compute_all(y, y_pred, top5, names555, species555, orders555,
                                       os.path.join(zs_env.RESULTS_DIR, run), run,
                                       extra=extra)
                else:
                    s = {'run': run, 'model': mname, 'level': level, 'space': space,
                         'taxo': taxo, 'n_test': int(len(y)), 'n_classes': int(tf.shape[0]),
                         'top1_accuracy': float((y_pred == y).mean()),
                         'top5_accuracy': float(top5.mean())}
                    os.makedirs(os.path.join(zs_env.RESULTS_DIR, run), exist_ok=True)
                    with open(os.path.join(zs_env.RESULTS_DIR, run, 'summary.json'),
                              'w', encoding='utf-8') as f:
                        json.dump(s, f, indent=2)
                rows.append(s)
                print(f'{run:34s} top1={100*s["top1_accuracy"]:6.2f}  '
                      f'top5={100*s["top5_accuracy"]:6.2f}'
                      + (f'  @404sp={100*s["species_404_accuracy"]:6.2f}'
                         f'  ntext={s["n_unique_texts"]:3d}'
                         f'  oracle-variant={100*s["oracle_variant_acc"]:6.2f}'
                         f' (prior {100*s["oracle_variant_prior"]:.2f},'
                         f' {100*s["oracle_variant_gain"]:+.2f})'
                         if space == 'leaf' else ''), flush=True)
        del model
        torch.cuda.empty_cache()

    import pandas as pd
    out = os.path.join(zs_env.RESULTS_DIR, 'zeroshot_summary.csv')
    pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8')
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
