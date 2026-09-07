#!/usr/bin/env bash
# Pass 5: đóng nốt hai trục mà pass 1-4 để hở.
#
#   bash src/run_more2.sh
#
# TRỤC THỜI GIAN KIẾN TRÚC — sau pass 5 thì mọi mốc lớn đều có mặt, cùng IN1k @224:
#   2012 AlexNet -> 2014 GoogLeNet/VGG -> 2015 ResNet -> 2017 DenseNet
#   -> 2019 EfficientNet -> 2019 MobileNetV3 -> 2022 ConvNeXt
#   AlexNet còn trả lời một câu riêng: kiến trúc CŨ + có pretrain so với
#   `cnn_scratch` (5M, kiến trúc mới hơn nhưng KHÔNG pretrain, 60.72) thì bên nào
#   hơn? Tức đòn bẩy pretrain có bù được 10 năm tiến bộ kiến trúc không.
#
# TRỤC QUY MÔ TRONG HỌ ViT — `vit_small_in21k` (21.9M) dùng ĐÚNG checkpoint family
#   `augreg_in21k` của `vit_b_16_in21k` (86.2M) và đúng recipe SGD 1e-3/1e-2, nên
#   chênh lệch quy hết về SỐ THAM SỐ. Đồng thời cho điểm so ba chiều ở ~22-28M
#   cùng pretrain 21k/22k: ViT-S vs Swin-T vs ConvNeXt-T.
#
# TRỤC HIỆU QUẢ — `efficientnet_b0` (4.7M, 2019) vs `mobilenet_v3_large` (4.9M,
#   2019): hai thiết kế "nhỏ mà tốt" cùng năm, cùng cỡ, cùng IN1k.
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/logs

. src/sweep_lock.sh
sweep_lock more2

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

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

run googlenet_224        googlenet        30 96 8
run efficientnet_b0_224  efficientnet_b0  30 64 8
run vit_small_in21k_224  vit_small_in21k  25 64 8
run alexnet_224          alexnet          30 128 8

# Ensemble chạy lại: cache có tăng dần nên chỉ tính thêm run mới.
echo "=== $(date +%H:%M:%S) START ensemble + TTA (lan 2, gom ca pass 5)"
python src/ensemble.py --cache --report > runs/logs/ensemble.log 2>&1
rc=$?                # phai lay NGAY: $(date) trong echo se ghi de $?
echo "=== $(date +%H:%M:%S) DONE  ensemble (exit $rc)"
tail -n 40 runs/logs/ensemble.log

echo "=== $(date +%H:%M:%S) MORE2 DONE"
python src/report.py --save
