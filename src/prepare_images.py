"""Pre-resize NABirds images to a smaller short side.

Ảnh gốc có median 1024x683 -> JPEG decode là bottleneck của DataLoader.
Resize offline một lần về cạnh ngắn 448 giúp epoch nhanh hơn nhiều mà không mất
thông tin cho mọi độ phân giải train <= 448 (chỉ 0.09% ảnh có cạnh ngắn < 224).

Bounding box KHÔNG được ghi lại ở đây; dataset sẽ tự scale bbox theo tỉ lệ
width_resized / width_goc khi load (xem nabirds_data.py).
"""
import os
import sys
from multiprocessing import Pool

from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nabirds')
SRC_DIR = os.path.join(ROOT, 'images')
SHORT_SIDE = int(os.environ.get('NAB_SHORT_SIDE', 448))
DST_DIR = os.path.join(ROOT, f'images_r{SHORT_SIDE}')
QUALITY = 92
WORKERS = int(os.environ.get('NAB_WORKERS', min(12, os.cpu_count() or 4)))


def resize_one(rel_path):
    src = os.path.join(SRC_DIR, rel_path)
    dst = os.path.join(DST_DIR, rel_path)
    if os.path.exists(dst):
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with Image.open(src) as im:
        im = im.convert('RGB')
        w, h = im.size
        s = SHORT_SIDE / min(w, h)
        if s < 1.0:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        im.save(dst, 'JPEG', quality=QUALITY)
    return 1


def main():
    rels = []
    with open(os.path.join(ROOT, 'images.txt'), encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rels.append(line.split(' ', 1)[1].replace('/', os.sep))
    print(f'{len(rels)} images -> {DST_DIR} (short side {SHORT_SIDE})', flush=True)
    done = 0
    with Pool(processes=WORKERS) as pool:
        for i, _ in enumerate(pool.imap_unordered(resize_one, rels, chunksize=64), 1):
            done = i
            if i % 4000 == 0:
                print(f'  {i}/{len(rels)}', flush=True)
    print(f'done {done}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
