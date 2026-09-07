#!/usr/bin/env bash
# Pass 8 — decoder "part query" (encoder-decoder + cross-attention).
#
#   bash src/run_partq.sh
#
# Bốn run, xếp theo chi phí. Mỗi run trả lời một câu khác nhau:
#
# | run | encoder | trả lời câu gì |
# |-----|---------|----------------|
# | frozen   | checkpoint NABirds, ĐÓNG BĂNG | decoder một mình đáng bao nhiêu? Rẻ nhất,
# |          |                               | và cho ngay bản đồ attention để đối chiếu keypoint |
# | ft       | checkpoint NABirds, mở khoá   | bản đầy đủ. So với 85.86 của chính encoder đó |
# | imagenet | ImageNet, KHÔNG qua NABirds   | đối chứng kiểu INTR. Biết trước là kém —
# |          |                               | INTR đạt CUB 71.8 so với 83.8 của ResNet-50 thường |
# | k24      | như `ft` nhưng K=24 bộ phận   | số bộ phận có quan trọng không (NABirds có 11 keypoint) |
#
# Vì sao encoder là ViT chứ không ConvNeXt: ConvNeXt-T ở 224 chỉ cho 49 token
# (7x7) — quá thô để tách bộ phận. ViT-B/16 cho 196 token (14x14).
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/logs

. src/sweep_lock.sh
sweep_lock partq

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

CKPT=runs/vit_b_16_in21k_224/best.pt

run () {                     # run <run_name> [extra...]
  local rname=$1; shift
  local log="runs/logs/${rname}.log"
  if [ -f "results/${rname}/summary.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "=== $(date +%H:%M:%S) BO QUA $rname"; return 0
  fi
  echo "=== $(date +%H:%M:%S) START $rname -> $log"
  python src/partquery.py --model vit_b_16_in21k --run-name "$rname" \
    --workers 10 "$@" > "$log" 2>&1
  local rc=$?          # phai lay NGAY: $(date) trong echo se ghi de $?
  echo "=== $(date +%H:%M:%S) DONE  $rname (exit $rc)"
  tail -n 10 "$log"
}

run partq_frozen   --encoder-ckpt "$CKPT" --freeze-encoder --epochs 15 --batch-size 64
run partq_ft       --encoder-ckpt "$CKPT" --epochs 15 --batch-size 48
run partq_imagenet --epochs 15 --batch-size 48
run partq_ft_k24   --encoder-ckpt "$CKPT" --epochs 15 --batch-size 48 --parts 24

echo "=== $(date +%H:%M:%S) PARTQ DONE"
python src/report.py --save
python src/aug_report.py
