#!/usr/bin/env bash
# Pass 6: các thí nghiệm augmentation trong PLAN mục E.
#
#   bash src/run_aug.sh
#
# Mọi run ở đây chỉ khác baseline ĐÚNG MỘT biến augmentation. Baseline đã có sẵn:
#   resnet50_224            (erasing 0.25, hue 0.02, rotate 0, không trộn)  78.61
#   convnext_tiny_in22k_224 (như trên, AdamW)                               85.55
#
# | nhóm | biến đổi | đối chứng với |
# |------|----------|---------------|
# | E1 | --erasing-p 0        | resnet50_224 / convnext_tiny_in22k_224 |
# | E2 | --rotate 15          | resnet50_224 |
# | E4 | --hue 0.10           | resnet50_224 |
# | E3 | --cutmix-alpha 1.0   | resnet50_224_ep100 (PHẢI có baseline 100 epoch riêng) |
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/logs

. src/sweep_lock.sh
sweep_lock aug

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

ADAMW="--optimizer adamw --lr-backbone 1e-4 --lr-head 1e-3 --weight-decay 0.05"

run () {                     # run <run_name> <model> <epochs> <batch> <patience> [extra...]
  local rname=$1 model=$2 epochs=$3 bs=$4 pat=$5; shift 5
  local log="runs/logs/${rname}.log"
  if [ -f "results/${rname}/summary.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "=== $(date +%H:%M:%S) BO QUA $rname (da co results/${rname}/summary.json)"
    return 0
  fi
  echo "=== $(date +%H:%M:%S) START $rname (epochs=$epochs bs=$bs patience=$pat) -> $log"
  python src/train.py --model "$model" --run-name "$rname" \
    --epochs "$epochs" --batch-size "$bs" --patience "$pat" \
    --val-frac 0.15 --workers 12 "$@" > "$log" 2>&1
  local rc=$?          # phai lay NGAY: $(date) trong echo se ghi de $?
  echo "=== $(date +%H:%M:%S) DONE  $rname (exit $rc)"
  tail -n 12 "$log"
}

# --- E1: RandomErasing co that su giup khong? -------------------------------
# Gia thuyet tu tai lieu: KHONG, va con hai. Cutout lam ResNet-50 mat ~2 diem
# tren CUB vi co xac suat xoa trung vung phan biet, ma vung do o chim rat nho.
run resnet50_224_noerase            resnet50            30 64 8 --erasing-p 0
run convnext_tiny_in22k_224_noerase convnext_tiny_in22k 25 64 8 $ADAMW --erasing-p 0

# --- E2: xoay 15 do --------------------------------------------------------
run resnet50_224_rot15              resnet50            30 64 8 --rotate 15

# --- E4: hue manh (DOI CHUNG cho D3, khong phai de doi mac dinh) -----------
# D3 noi mau bo long CHINH LA nhan. Neu run nay tut thi D3 duoc xac nhan bang so.
run resnet50_224_hue10              resnet50            30 64 8 --hue 0.10

# --- E3: mixup/cutmix CHI co nghia khi train dai ---------------------------
# Benchmark OpenMixup tren CUB do o 200 epoch. Chay o 30 epoch roi ket luan
# "mixup vo dung" la LOI DO. Nen phai co baseline 100 epoch RIENG de so.
# patience 20 (khong phai 8): mixup lam hoi tu cham, patience 8 se cat som oan.
run resnet50_224_ep100              resnet50           100 64 20
run resnet50_224_ep100_cutmix       resnet50           100 64 20 \
    --cutmix-alpha 1.0 --mix-prob 0.5

echo "=== $(date +%H:%M:%S) AUG DONE"
python src/report.py --save
python src/aug_report.py
