"""Gộp kết quả của mọi run trong `results/` thành bảng so sánh.

  python src/report.py                # bảng tổng hợp tất cả run
  python src/report.py --run resnet50_224   # chi tiết 1 run
"""
import argparse
import glob
import json
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, 'results')

COLS = [
    ('model', 'model'), ('img_size', 'img'), ('params_M', 'params(M)'),
    ('pretrained', 'pretrain'), ('epochs', 'ep'), ('train_minutes', 'min'),
    ('top1_accuracy', 'top1'), ('top5_accuracy', 'top5'),
    ('balanced_accuracy', 'bal-acc'),
    ('macro_precision', 'macro-P'), ('macro_recall', 'macro-R'), ('macro_f1', 'macro-F1'),
    ('micro_f1', 'micro-F1'), ('weighted_f1', 'w-F1'),
    ('species_404_accuracy', 'acc@404sp'), ('order_22_accuracy', 'acc@22ord'),
]
PCT = {'top1_accuracy', 'top5_accuracy', 'balanced_accuracy', 'macro_precision',
       'macro_recall', 'macro_f1', 'micro_f1', 'weighted_f1',
       'species_404_accuracy', 'order_22_accuracy'}


def load_all():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, '*', 'summary.json'))):
        with open(path, encoding='utf-8') as f:
            s = json.load(f)
        if s.get('run', '').startswith('_'):      # bỏ qua run smoke test
            continue
        row = {'run': s.get('run')}
        for key, label in COLS:
            v = s.get(key)
            if v is not None and key in PCT:
                v = round(v * 100, 2)
            row[label] = v
        row['bbox'] = s.get('use_bbox')
        row['err_same_species%'] = round(s.get('errors_within_same_species_pct', float('nan')), 1)
        row['err_same_order%'] = round(s.get('errors_within_same_order_pct', float('nan')), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def detail(run):
    d = os.path.join(RESULTS, run)
    pc = pd.read_csv(os.path.join(d, 'per_class.csv'))
    print(f'\n### {run} — 15 lớp KÉM nhất (theo F1)')
    print(pc.head(15)[['class_name', 'order', 'support', 'precision', 'recall',
                       'f1', 'top5_recall', 'most_confused_with']].to_string(index=False))
    print(f'\n### {run} — 10 lớp TỐT nhất')
    print(pc.tail(10)[['class_name', 'support', 'precision', 'recall', 'f1']]
          .iloc[::-1].to_string(index=False))
    po = pd.read_csv(os.path.join(d, 'per_order.csv'))
    print(f'\n### {run} — theo order (22)')
    print(po.to_string(index=False))
    cf = os.path.join(d, 'confusions.csv')
    if os.path.exists(cf):
        print(f'\n### {run} — 20 cặp nhầm nhiều nhất')
        print(pd.read_csv(cf).head(20).to_string(index=False))
    print(f'\n### {run} — F1 theo nhóm support')
    pc['bucket'] = pd.cut(pc['support'], [0, 20, 30, 40, 50, 61],
                          labels=['<=20', '21-30', '31-40', '41-50', '51-60'])
    print(pc.groupby('bucket', observed=True)
          .agg(n_classes=('f1', 'size'), mean_f1=('f1', 'mean'),
               mean_recall=('recall', 'mean'), mean_precision=('precision', 'mean'))
          .round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=None)
    ap.add_argument('--save', action='store_true', help='ghi results/comparison.csv|.md')
    a = ap.parse_args()
    if a.run:
        detail(a.run)
        return
    df = load_all()
    if df.empty:
        print('chưa có run nào trong results/')
        return
    df = df.sort_values('top1', ascending=False)
    print(df.to_string(index=False))
    if a.save:
        df.to_csv(os.path.join(RESULTS, 'comparison.csv'), index=False, encoding='utf-8-sig')
        with open(os.path.join(RESULTS, 'comparison.md'), 'w', encoding='utf-8') as f:
            f.write(df.to_markdown(index=False))
        print('\n-> results/comparison.csv, results/comparison.md')


if __name__ == '__main__':
    main()
