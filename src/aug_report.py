"""Báo cáo cho các thí nghiệm augmentation (PLAN mục E.2).

Top-1 tổng KHÔNG đủ để đánh giá augmentation nhắm vào lớp hiếm: 113 lớp có <30
ảnh train chỉ chiếm ~10% ảnh test, nên một cải thiện thật ở đó vẫn có thể chìm
trong sai số của top-1. Script này tách bốn con số:

  1. top-1 tổng                       — để chắc không đánh đổi ngược
  2. macro-F1 trên 113 lớp ĐUÔI       — mục tiêu chính của E1/E2
  3. err_same_species%                — nhầm trong cùng loài
  4. recall trên các CẶP LOÀI GẦN GIỐNG — chỗ hỏng nặng nhất hiện nay

Định nghĩa tập "lớp khó" — CHỖ DỄ SAI NHẤT:
  Bản đầu tôi lấy danh sách này từ `results/bioclip_224/per_class.csv` rồi so
  bioclip trên chính nó. Đó là **selection bias**: chọn lớp VÌ model đó thua,
  rồi kết luận model đó thua. Bản này định nghĩa tập khó **độc lập với mọi model
  đơn**: lấy F1 TRUNG BÌNH của mỗi lớp trên TOÀN BỘ run đã có, rồi giữ N lớp
  thấp nhất. Tập tính một lần, dùng chung cho mọi so sánh về sau.

  `Northwestern Crow` bị loại thẳng: AOS đã GỘP nó vào `American Crow` năm 2020,
  nên đó là vấn đề NHÃN chứ không phải model — không augmentation nào sửa được.

    python src/aug_report.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, 'results')

# Lớp có ít hơn ngưỡng này ảnh TRAIN được coi là lớp đuôi.
TAIL_TRAIN_MAX = 30

N_HARD = 20
# Lớp mà chính giới phân loại học đã gộp -> lỗi NHÃN, không phải lỗi model.
EXCLUDE = {'Northwestern Crow'}


def tail_classes():
    """Tên các lớp có < TAIL_TRAIN_MAX ảnh TRAIN (đọc từ split gốc, không phải test)."""
    import collections
    from nabirds_data import NABirds
    ds = NABirds(split='train')
    sup = collections.Counter(lab for _, _, lab in ds.samples)
    names = ds.idx_to_name
    return {names[c] for c in range(len(names)) if sup[c] < TAIL_TRAIN_MAX}


def hard_classes(paths):
    """N lớp có F1 TRUNG BÌNH thấp nhất trên toàn bộ run — không lệ thuộc model nào."""
    acc = {}
    for p in paths:
        pc = pd.read_csv(p)
        for name, f1 in zip(pc['class_name'], pc['f1']):
            acc.setdefault(name, []).append(f1)
    mean_f1 = {k: sum(v) / len(v) for k, v in acc.items() if k not in EXCLUDE}
    return set(sorted(mean_f1, key=mean_f1.get)[:N_HARD]), mean_f1


def main():
    tail = tail_classes()
    pc_paths = [p for p in sorted(glob.glob(os.path.join(RESULTS, '*', 'per_class.csv')))
                if not os.path.basename(os.path.dirname(p)).startswith('_')]
    hard_set, mean_f1 = hard_classes(pc_paths)
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, '*', 'summary.json'))):
        run = os.path.basename(os.path.dirname(path))
        pc_path = os.path.join(os.path.dirname(path), 'per_class.csv')
        if run.startswith('_') or not os.path.exists(pc_path):
            continue
        s = json.load(open(path, encoding='utf-8'))
        pc = pd.read_csv(pc_path)
        is_tail = pc['class_name'].isin(tail)
        hard = pc[pc['class_name'].isin(hard_set)]
        rows.append({
            'run': run,
            'top1': round(s['top1_accuracy'] * 100, 2),
            'macroF1': round(s['macro_f1'] * 100, 2),
            f'F1@tail<{TAIL_TRAIN_MAX}': round(pc.loc[is_tail, 'f1'].mean() * 100, 2),
            'n_tail': int(is_tail.sum()),
            'F1@head': round(pc.loc[~is_tail, 'f1'].mean() * 100, 2),
            'recall@hard': round(hard['recall'].mean() * 100, 2) if len(hard) else None,
            'n_hard': len(hard),
            'errSameSp%': round(s.get('errors_within_same_species_pct', float('nan')), 1),
        })
    if not rows:
        print('chua co run nao co per_class.csv')
        return
    df = pd.DataFrame(rows).sort_values('top1', ascending=False)
    print(f'\n{len(tail)} lop co < {TAIL_TRAIN_MAX} anh train.')
    print(f'{len(hard_set)} lop KHO = F1 trung binh thap nhat tren {len(pc_paths)} run '
          f'(khong lay tu mot model nao):')
    for n in sorted(hard_set, key=mean_f1.get):
        print(f'    {mean_f1[n]*100:5.1f}  {n}')
    print()
    print(df.to_string(index=False))
    out = os.path.join(RESULTS, 'aug_comparison')
    df.to_csv(out + '.csv', index=False, encoding='utf-8-sig')
    with open(out + '.md', 'w', encoding='utf-8') as f:
        f.write(df.to_markdown(index=False))
    print(f'\n-> results/aug_comparison.csv|.md')


if __name__ == '__main__':
    main()
