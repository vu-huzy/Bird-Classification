"""Thiết lập môi trường cho thí nghiệm zero-shot. **Import module này TRƯỚC** khi
import `transformers` / `open_clip` / `torch`.

Hai việc:

1. `USE_TF=0` — máy này có TensorFlow build theo NumPy 1.x nhưng NumPy hiện tại là
   2.4.6. `transformers.image_transforms` gọi `import tensorflow` khi
   `is_tf_available()` -> `ImportError: numpy.core.umath failed to import`, kéo theo
   `import peft` chết. Tắt nhánh TF là đủ, KHÔNG gỡ TensorFlow của môi trường.

2. `TORCH_HOME` / `HF_HOME` trỏ vào `<repo>/models` — giống `src/models_zoo.py`, để
   mọi checkpoint nằm gọn trong repo.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(REPO, 'models')
ZS_DIR = os.path.join(REPO, 'zeroshot')
DATA_DIR = os.path.join(ZS_DIR, 'data')
CACHE_DIR = os.path.join(ZS_DIR, 'cache')
RESULTS_DIR = os.path.join(ZS_DIR, 'results')

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
os.environ.setdefault('TORCH_HOME', MODEL_DIR)
os.environ.setdefault('HF_HOME', os.path.join(MODEL_DIR, 'hf'))

for d in (DATA_DIR, CACHE_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

# Console Windows mặc định cp1252 -> in tên loài có dấu (vd 'Aigle royal') là crash.
import sys  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')
