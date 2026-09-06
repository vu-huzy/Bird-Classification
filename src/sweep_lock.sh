# Khoa chong chay chong cho cac script sweep.  Dung bang `source`:
#
#     source src/sweep_lock.sh <ten>      # vd: all, 448, full
#
# Moi script giu MOT PID file rieng (runs/.sweep_<ten>.pid). Ly do khong dung
# chung mot file: `run_full.sh` goi `run_all.sh` nen ca hai cung song, mot file
# duy nhat se khien script con tuong la co sweep khac dang chay.
#
# Vi sao khong so khop command line qua WMI: cach do tung giet nham chinh shell
# dang goi script, vi command line cua no cung chua chuoi "run_all".
sweep_lock() {
  local name=$1
  mkdir -p runs
  SWEEP_PIDFILE="runs/.sweep_${name}.pid"
  if [ -f "$SWEEP_PIDFILE" ]; then
    local other
    other=$(cat "$SWEEP_PIDFILE" 2>/dev/null || echo "")
    if [ -n "$other" ] && kill -0 "$other" 2>/dev/null; then
      echo "LOI: '$name' dang chay roi (bash PID $other)."
      echo "     Dung sach truoc bang:  bash src/stop_all.sh"
      exit 1
    fi
  fi
  echo $$ > "$SWEEP_PIDFILE"
  trap 'rm -f "$SWEEP_PIDFILE"' EXIT
}
