#!/usr/bin/env bash
# Chạy tuần tự toàn bộ model trên NABirds 555 lớp (giao thức chuẩn: split gốc,
# không dùng bbox lúc test). Log riêng cho từng run trong runs/logs/.
#
#   bash src/run_all.sh
#
# Ghi chú 1: KHÔNG dùng `| tee` — tee block-buffer khi ghi ra file, làm tiến độ
#            không hiện ra cho tới khi process kết thúc. Redirect thẳng vào file.
# Ghi chú 2: checkpoint tốt nhất và early stopping chọn theo tập VAL tách từ
#            train (--val-frac 0.15 -> 3,597 ảnh, phủ đủ 555 lớp, sai số
#            chuẩn của val top-1 ~+-0.72%). Tập test chỉ chạy một lần ở cuối.
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/logs

. src/sweep_lock.sh
sweep_lock all

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

run () {                     # run <model> <epochs> <batch> <patience> [extra...]
  local model=$1 epochs=$2 bs=$3 pat=$4; shift 4
  local log="runs/logs/${model}.log"
  local rname="${model}_224"
  [ "$model" = "inception_v3" ] && rname="${model}_299"
  # Resume: bo qua model da co ket qua day du. Cho phep sua loi va chay tiep
  # ma khong phai train lai tu dau nhung model da xong.
  if [ -f "results/${rname}/summary.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "=== $(date +%H:%M:%S) BO QUA $model (da co results/${rname}/summary.json)"
    return 0
  fi
  echo "=== $(date +%H:%M:%S) START $model (epochs=$epochs bs=$bs patience=$pat) -> $log"
  python src/train.py --model "$model" --epochs "$epochs" --batch-size "$bs"     --patience "$pat" --val-frac 0.15 --workers 12 "$@" > "$log" 2>&1
  local rc=$?
  echo "=== $(date +%H:%M:%S) DONE  $model (exit $rc)"
  tail -n 14 "$log"
}

run cnn_scratch    60 128 12 --lr-head 0.05
run resnet50       30  64  8
run resnet101      30  48  8
# inception_v3: do lap nhieu lan cho thay bs48 cham hon han bs96
# (TB 166 vs 233 img/s). Batch x2 -> scale LR x2 theo linear scaling rule.
run inception_v3   30  96  8 --lr-backbone 0.01 --lr-head 0.1
run vit_b_16       25  64  8
run vit_b_16_in21k 25  64  8

echo "=== $(date +%H:%M:%S) ALL DONE"
python src/report.py --save
