#!/usr/bin/env bash
# Pre-flight: chay 8 step train + eval day du + xuat metrics cho TUNG model.
# Muc dich la bat loi runtime (aux logits cua Inception, ten head cua ViT, ghi
# CSV, ...) trong ~10 phut thay vi phat hien sau 3 tieng chay that.
#
#   bash src/preflight.sh
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/preflight
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

MODELS="cnn_scratch resnet50 resnet101 inception_v3 vit_b_16 vit_b_16_in21k"
fail=0
for m in $MODELS; do
  log="runs/preflight/${m}.log"
  printf '%-16s ... ' "$m"
  python src/train.py --model "$m" --epochs 1 --max-steps 8 --batch-size 32 \
    --workers 6 --val-frac 0.15 --patience 0 --run-name "_pf_${m}" > "$log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "LOI (exit $rc)"
    tail -n 15 "$log"
    fail=1
  else
    top1=$(grep -m1 'top-1 accuracy' "$log" | awk '{print $3}')
    ntr=$(grep -m1 '  train ' "$log" | sed 's/.*train \([0-9]*\).*/\1/')
    echo "OK   top-1=${top1:-?}  (chi 8 step nen thap la dung)"
  fi
done

rm -rf runs/_pf_* results/_pf_*
if [ $fail -eq 0 ]; then
  echo "PRE-FLIGHT: tat ca 6 model chay duoc."
else
  echo "PRE-FLIGHT: CO MODEL LOI — xem runs/preflight/*.log"
fi
exit $fail
