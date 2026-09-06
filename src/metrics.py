"""Tính và xuất bảng kết quả chi tiết cho NABirds 555 lớp.

Xuất ra:
  per_class.csv   555 dòng: support, TP/FP/FN, precision, recall, f1, top-5 recall
  per_order.csv   22 dòng gộp theo order
  per_species.csv 404 dòng gộp theo species (gộp biến thể giới tính/tuổi)
  confusions.csv  các cặp bị nhầm nhiều nhất
  summary.json    toàn bộ chỉ số tổng hợp

Lưu ý về micro trong bài single-label multiclass: micro-P = micro-R = micro-F1
= top-1 accuracy (mỗi mẫu đóng góp đúng 1 dự đoán). Vẫn xuất đủ để đối chiếu.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, precision_recall_fscore_support)


def _agg(y_true, y_pred, groups):
    """Gộp nhãn lá về một mức cao hơn rồi tính accuracy."""
    g = np.asarray(groups)
    return accuracy_score(g[y_true], g[y_pred])


def compute_all(y_true, y_pred, top5_correct, class_names, species, orders,
                out_dir, run_name, extra=None):
    """y_true/y_pred: (N,) int64 trong [0,554]. top5_correct: (N,) bool."""
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(class_names)
    labels = np.arange(n_classes)

    # ---------------- tổng hợp ----------------
    top1 = accuracy_score(y_true, y_pred)
    top5 = float(np.mean(top5_correct))
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0)
    p_mi, r_mi, f_mi, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='micro', zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0)

    summary = {
        'run': run_name,
        'n_test': int(len(y_true)),
        'n_classes': n_classes,
        'top1_accuracy': top1,
        'top5_accuracy': top5,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'macro_precision': p_ma, 'macro_recall': r_ma, 'macro_f1': f_ma,
        'micro_precision': p_mi, 'micro_recall': r_mi, 'micro_f1': f_mi,
        'weighted_precision': p_w, 'weighted_recall': r_w, 'weighted_f1': f_w,
        'species_404_accuracy': _agg(y_true, y_pred, species),
        'order_22_accuracy': _agg(y_true, y_pred, orders),
    }
    if extra:
        summary.update(extra)

    # ---------------- per-class (555 dòng) ----------------
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    tp = np.array([np.sum((y_true == c) & (y_pred == c)) for c in labels])
    fp = np.array([np.sum((y_true != c) & (y_pred == c)) for c in labels])
    fn = sup - tp
    top5_per_class = np.array([
        top5_correct[y_true == c].mean() if np.any(y_true == c) else 0.0 for c in labels])
    # lớp bị nhầm sang nhiều nhất
    worst_into = []
    for c in labels:
        mask = (y_true == c) & (y_pred != c)
        if mask.sum() == 0:
            worst_into.append('')
            continue
        vals, cnts = np.unique(y_pred[mask], return_counts=True)
        worst_into.append(f'{class_names[vals[cnts.argmax()]]} ({cnts.max()})')

    per_class = pd.DataFrame({
        'class_idx': labels,
        'class_name': class_names,
        'species': species,
        'order': orders,
        'support': sup, 'TP': tp, 'FP': fp, 'FN': fn,
        'precision': p, 'recall': r, 'f1': f,
        'accuracy': r,                       # với 1 lớp, accuracy = recall
        'top5_recall': top5_per_class,
        'most_confused_with': worst_into,
    }).sort_values('f1')
    per_class.to_csv(os.path.join(out_dir, 'per_class.csv'), index=False,
                     encoding='utf-8-sig', float_format='%.4f')

    # ---------------- gộp theo order / species ----------------
    def group_table(col):
        df = per_class.copy()
        rows = []
        for g, sub in df.groupby(col, sort=False):
            s = sub['support'].sum()
            rows.append({
                col: g, 'n_classes': len(sub), 'support': int(s),
                'accuracy': sub['TP'].sum() / s if s else 0.0,
                'macro_precision': sub['precision'].mean(),
                'macro_recall': sub['recall'].mean(),
                'macro_f1': sub['f1'].mean(),
                'top5_recall': float(np.average(sub['top5_recall'], weights=sub['support'])),
            })
        return pd.DataFrame(rows).sort_values('accuracy')

    group_table('order').to_csv(
        os.path.join(out_dir, 'per_order.csv'), index=False,
        encoding='utf-8-sig', float_format='%.4f')
    group_table('species').to_csv(
        os.path.join(out_dir, 'per_species.csv'), index=False,
        encoding='utf-8-sig', float_format='%.4f')

    # ---------------- cặp nhầm nhiều nhất ----------------
    wrong = y_true != y_pred
    if wrong.any():
        pairs, cnts = np.unique(
            np.stack([y_true[wrong], y_pred[wrong]]), axis=1, return_counts=True)
        order_idx = np.argsort(-cnts)[:60]
        conf = pd.DataFrame({
            'true_class': [class_names[i] for i in pairs[0, order_idx]],
            'pred_class': [class_names[i] for i in pairs[1, order_idx]],
            'count': cnts[order_idx],
            'same_species': [species[i] == species[j]
                             for i, j in zip(pairs[0, order_idx], pairs[1, order_idx])],
            'same_order': [orders[i] == orders[j]
                           for i, j in zip(pairs[0, order_idx], pairs[1, order_idx])],
        })
        conf.to_csv(os.path.join(out_dir, 'confusions.csv'), index=False,
                    encoding='utf-8-sig')
        n_err = int(wrong.sum())
        sp = np.asarray(species)
        od = np.asarray(orders)
        summary['errors_total'] = n_err
        summary['errors_within_same_species_pct'] = float(
            np.mean(sp[y_true[wrong]] == sp[y_pred[wrong]]) * 100)
        summary['errors_within_same_order_pct'] = float(
            np.mean(od[y_true[wrong]] == od[y_pred[wrong]]) * 100)

    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary, per_class


def format_summary(s):
    L = [
        f"top-1 accuracy      {s['top1_accuracy']*100:6.2f}%",
        f"top-5 accuracy      {s['top5_accuracy']*100:6.2f}%",
        f"balanced accuracy   {s['balanced_accuracy']*100:6.2f}%",
        f"macro    P/R/F1     {s['macro_precision']*100:5.2f} / {s['macro_recall']*100:5.2f} / {s['macro_f1']*100:5.2f}",
        f"micro    P/R/F1     {s['micro_precision']*100:5.2f} / {s['micro_recall']*100:5.2f} / {s['micro_f1']*100:5.2f}",
        f"weighted P/R/F1     {s['weighted_precision']*100:5.2f} / {s['weighted_recall']*100:5.2f} / {s['weighted_f1']*100:5.2f}",
        f"accuracy @404 species {s['species_404_accuracy']*100:6.2f}%",
        f"accuracy @22 order    {s['order_22_accuracy']*100:6.2f}%",
    ]
    if 'errors_within_same_species_pct' in s:
        L.append(f"lỗi trong cùng loài  {s['errors_within_same_species_pct']:5.2f}%  "
                 f"| trong cùng order {s['errors_within_same_order_pct']:5.2f}%")
    return '\n'.join('  ' + x for x in L)
