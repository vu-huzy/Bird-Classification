#!/usr/bin/env bash
# Dung SACH moi tien trinh cua sweep.
#
#   bash src/stop_all.sh
#
# Vi sao can: dung job tu ben ngoai (Ctrl-C, kill task wrapper) chi giet process
# vo. Process `bash run_all.sh` van song, chay tiep sang model ke tiep, va
# DataLoader worker cu tro thanh mo coi. Da tung co 7 sweep chay song song voi
# 34 process python, lam RAM tut con duoi 1 GB va moi so do toc do vo nghia.
set -u
cd "$(dirname "$0")/.."

# 1) Giet bash sweep theo PID file (moi script co file rieng).
shopt -s nullglob
found=0
for f in runs/.sweep_*.pid; do
  pid=$(cat "$f" 2>/dev/null || echo "")
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "dung $(basename "$f" .pid) -> bash PID $pid"
    kill -TERM "$pid" 2>/dev/null || true
    found=1
  fi
  rm -f "$f"
done
[ "$found" = "0" ] && echo "khong co bash sweep nao dang chay"
sleep 1
for f in runs/.sweep_*.pid; do rm -f "$f"; done

# 2) Worker python KHONG chet theo bash cha tren Windows -> phai giet rieng.
#    Day moi la phan quan trong: chinh chung an het RAM.
powershell -NoProfile -Command '
  $p = @(Get-Process python -ErrorAction SilentlyContinue)
  "python dang chay: " + $p.Count
  $p | Stop-Process -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 3
  $os = Get-CimInstance Win32_OperatingSystem
  "--- sau khi don --- python={0}  FreeMB={1}" -f (Get-Process python -ea 0).Count, [int]($os.FreePhysicalMemory/1KB)
'
