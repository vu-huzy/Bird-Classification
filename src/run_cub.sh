#!/usr/bin/env bash
# Pass 7 — C3: chuyển giao NABirds -> CUB-200-2011.
#
#   bash src/run_cub.sh
#
# Mỗi model chạy HAI nhánh khác nhau đúng một biến: trọng số khởi tạo backbone.
#   imagenet  : đi thẳng ImageNet -> CUB
#   nabirds   : ImageNet -> NABirds (555 lớp) -> CUB
# Cùng recipe, cùng epoch, cùng lr, cùng augmentation.
#
# Đọc kết quả phải kèm: 142/200 loài CUB có mặt trong NABirds (docs/02 mục 0),
# nên đây KHÔNG phải transfer sang miền mới. Nó đo "thấy thêm ảnh chim có giúp
# không", và phần lớn lợi ích có thể đến từ 142 loài trùng đó.
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/logs

. src/sweep_lock.sh
sweep_lock cub

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

ADAMW="--optimizer adamw --lr-backbone 1e-4 --lr-head 1e-3 --weight-decay 0.05"

run () {                     # run <model> <init> <ckpt|-> [extra...]
  local model=$1 init=$2 ckpt=$3; shift 3
  local rname="${model}_224_${init}"
  local log="runs/logs/cub_${rname}.log"
  if [ -f "results/cub/${rname}/summary.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "=== $(date +%H:%M:%S) BO QUA cub/$rname"; return 0
  fi
  local ck=""
  [ "$ckpt" != "-" ] && ck="--ckpt $ckpt"
  echo "=== $(date +%H:%M:%S) START cub/$rname -> $log"
  python src/cub_transfer.py --model "$model" --init "$init" $ck \
    --epochs 25 --batch-size 64 --workers 8 "$@" > "$log" 2>&1
  local rc=$?          # phai lay NGAY: $(date) trong echo se ghi de $?
  echo "=== $(date +%H:%M:%S) DONE  cub/$rname (exit $rc)"
  tail -n 6 "$log"
}

run vit_b_16_in21k      imagenet -
run vit_b_16_in21k      nabirds  runs/vit_b_16_in21k_224/best.pt
run convnext_tiny_in22k imagenet - $ADAMW
run convnext_tiny_in22k nabirds  runs/convnext_tiny_in22k_224/best.pt $ADAMW
# BioCLIP: nguon pretrain sinh hoc + chang trung gian NABirds, nhanh manh nhat.
run bioclip             imagenet -
run bioclip             nabirds  runs/bioclip_224/best.pt

echo "=== $(date +%H:%M:%S) CUB DONE"
python src/cub_transfer.py --report
