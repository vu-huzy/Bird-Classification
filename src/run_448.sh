#!/usr/bin/env bash
# Pass thứ 2: train ở 448 để đo hiệu ứng độ phân giải.
# Chỉ chạy cnn_scratch và resnet50 (theo quyết định), dùng cache ảnh cạnh ngắn
# 512 để eval transform Resize(511)+CenterCrop(448) KHÔNG phải phóng to ảnh.
#
# Chạy sau khi src/run_all.sh xong:
#   NAB_SHORT_SIDE=512 python src/prepare_images.py   # nếu chưa có images_r512
#   bash src/run_448.sh
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/logs
. src/sweep_lock.sh
sweep_lock 448

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

run () {                     # run <model> <epochs> <batch> <patience> [extra...]
  local model=$1 epochs=$2 bs=$3 pat=$4; shift 4
  local log="runs/logs/${model}_448.log"
  local rname="${model}_448"
  # Resume: bo qua model da co ket qua day du. Cho phep sua loi va chay tiep
  # ma khong phai train lai tu dau nhung model da xong.
  if [ -f "results/${rname}/summary.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "=== $(date +%H:%M:%S) BO QUA $model (da co results/${rname}/summary.json)"
    return 0
  fi
  echo "=== $(date +%H:%M:%S) START ${model}@448 (epochs=$epochs bs=$bs patience=$pat) -> $log"
  python src/train.py --model "$model" --img-size 448 --image-dir images_r512 \
    --epochs "$epochs" --batch-size "$bs" --patience "$pat" \
    --val-frac 0.15 --workers 6 "$@" > "$log" 2>&1
  local rc=$?
  echo "=== $(date +%H:%M:%S) DONE  ${model}@448 (exit $rc)"
  tail -n 14 "$log"
}

# batch size lấy từ số đo thực trên RTX 5070 12GB:
#   cnn_scratch @448 bs48 -> 448 img/s, peak 3.44 GB
#   resnet50    @448 bs32 -> 200 img/s, peak 5.71 GB
run cnn_scratch 60 48 12 --lr-head 0.05
run resnet50    30 32  8

echo "=== $(date +%H:%M:%S) 448 DONE"
python src/report.py --save
