#!/usr/bin/env bash
# Xem nhanh trang thai sweep dang chay.
#
#   bash src/status.sh
set -u
cd "$(dirname "$0")/.."

echo "===== $(date '+%Y-%m-%d %H:%M:%S') ====="

# --- tien do tong the ---
echo
echo "--- Lich chay ---"
if [ -f runs/sweep.log ]; then
  grep -E "^(===|####)" runs/sweep.log | tail -14
else
  echo "  (chua co runs/sweep.log — sweep chua chay)"
fi

# --- model dang chay ---
cur=$(ls -t runs/logs/*.log 2>/dev/null | head -1)
if [ -n "${cur:-}" ]; then
  echo
  echo "--- Log moi nhat: $(basename "$cur" .log) ---"
  sed -n '2,3p' "$cur"
  echo "  epoch gan nhat:"
  grep -E "^  ep " "$cur" | tail -3 | sed 's/^/  /'
  echo "  step gan nhat:"
  grep -E "\] step " "$cur" | tail -2 | sed 's/^/  /'
  grep "early stopping" "$cur" 2>/dev/null | sed 's/^/  /'
fi

# --- ket qua da xong ---
echo
echo "--- Ket qua da hoan tat ---"
if ls results/*/summary.json >/dev/null 2>&1; then
  python - <<'PY'
import glob, json, os
rows = []
for p in sorted(glob.glob('results/*/summary.json')):
    if os.path.basename(os.path.dirname(p)).startswith('_'):
        continue
    with open(p, encoding='utf-8') as f:
        s = json.load(f)
    rows.append((s.get('run', '?'), s.get('top1_accuracy', 0) * 100,
                 s.get('top5_accuracy', 0) * 100, s.get('macro_f1', 0) * 100,
                 s.get('epochs_run', 0), s.get('best_epoch', 0),
                 s.get('train_minutes', 0)))
if rows:
    print('  %-22s %7s %7s %8s %5s %6s %7s' % ('run', 'top1', 'top5', 'macroF1', 'eps', 'best', 'phut'))
    for r in sorted(rows, key=lambda x: -x[1]):
        print('  %-22s %6.2f%% %6.2f%% %7.2f%% %5d %6d %7.1f' % r)
else:
    print('  (chua co)')
PY
else
  echo "  (chua co)"
fi

# --- tai nguyen ---
echo
echo "--- Tai nguyen ---"
powershell -NoProfile -Command '
  $p = @(Get-Process python -ErrorAction SilentlyContinue)
  $os = Get-CimInstance Win32_OperatingSystem
  "  python={0}  RAM_tu_do={1}MB" -f $p.Count, [int]($os.FreePhysicalMemory/1KB)
'
shopt -s nullglob
alive=0
for f in runs/.sweep_*.pid; do
  pid=$(cat "$f" 2>/dev/null || echo "")
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "  $(basename "$f" .pid): bash PID $pid (dang chay)"
    alive=$((alive+1))
  else
    echo "  CANH BAO: $(basename "$f" .pid) con PID file nhung tien trinh da chet"
  fi
done
[ "$alive" = "0" ] && echo "  khong co sweep nao dang chay"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /'
