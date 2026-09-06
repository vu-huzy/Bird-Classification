"""Lấy mẫu GPU / RAM / CPU mỗi giây trong lúc train, ghi ra CSV.

  python src/watch_resources.py --seconds 300 --out runs/resources.csv

Dùng để trả lời câu hỏi "GPU lúc cao lúc thấp có bất thường không" bằng số liệu
thay vì đọc nvidia-smi một lần. Cột `phase` suy ra từ log train đang chạy.
"""
import argparse
import csv
import os
import re
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SMI = ['nvidia-smi',
       '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw',
       '--format=csv,noheader,nounits']


def gpu_sample():
    try:
        out = subprocess.run(SMI, capture_output=True, text=True, timeout=5).stdout
        parts = [p.strip() for p in out.strip().split(',')]
        return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    except Exception:
        return float('nan'), float('nan'), float('nan'), float('nan')


PS = ('$os=Get-CimInstance Win32_OperatingSystem;'
      '$p=Get-Process python -ErrorAction SilentlyContinue;'
      '"{0},{1},{2}" -f [int]($os.FreePhysicalMemory/1KB),'
      '($p|Measure-Object).Count,[int](($p|Measure-Object WS -Sum).Sum/1MB)')


def host_sample():
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', PS],
                             capture_output=True, text=True, timeout=10).stdout
        free_mb, n_proc, ws_mb = out.strip().split(',')
        return int(free_mb), int(n_proc), int(ws_mb)
    except Exception:
        return -1, -1, -1


STEP_RE = re.compile(r'\[(\d+)\] step\s+(\d+)/(\d+)')


def tail_phase(log_path, last_size):
    """Đọc phần mới của log để biết đang ở step nào -> suy ra pha."""
    if not log_path or not os.path.exists(log_path):
        return 'unknown', last_size, ''
    size = os.path.getsize(log_path)
    if size == last_size:
        return 'eval_or_idle', size, ''      # log không tiến -> không ở giữa epoch
    with open(log_path, 'rb') as f:
        f.seek(max(0, size - 4096))
        chunk = f.read().decode('utf-8', 'replace')
    line = chunk.strip().split('\n')[-1]
    return ('train' if STEP_RE.search(line) else 'epoch_boundary'), size, line.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=int, default=300)
    ap.add_argument('--interval', type=float, default=1.0)
    ap.add_argument('--log', default=None, help='log của run đang chạy, để gắn nhãn pha')
    ap.add_argument('--out', default=os.path.join(REPO, 'runs', 'resources.csv'))
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t_end = time.time() + a.seconds
    last_size = -1
    n = 0
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['t', 'gpu_util', 'gpu_mem_util', 'gpu_mem_MB', 'power_W',
                    'free_ram_MB', 'n_python', 'python_ws_MB', 'phase'])
        t0 = time.time()
        while time.time() < t_end:
            gu, gmu, gm, pw = gpu_sample()
            free_mb, n_proc, ws_mb = host_sample()
            phase, last_size, _ = tail_phase(a.log, last_size)
            w.writerow([round(time.time() - t0, 1), gu, gmu, gm, pw,
                        free_mb, n_proc, ws_mb, phase])
            f.flush()
            n += 1
            time.sleep(a.interval)
    print(f'{n} mẫu -> {a.out}')


if __name__ == '__main__':
    main()
