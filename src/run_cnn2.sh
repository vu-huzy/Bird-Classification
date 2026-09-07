#!/usr/bin/env bash
# Pass 3: CNN HIỆN ĐẠI — mở hai trục mà pass 1/2 chưa chạm tới.
#
#   Trục KIẾN TRÚC   : resnet50(IN1k,224) vs convnext_tiny(IN1k,224)
#                      cùng nguồn pretrain, cùng độ phân giải, params 24.6 vs 28.2M
#                      -> chênh lệch quy được về thiết kế kiến trúc 2016 vs 2022.
#   Trục PRETRAIN    : convnext_tiny(IN1k) vs convnext_tiny(IN22k)
#                      cùng kiến trúc, chỉ đổi nguồn pretrain
#                      -> lặp lại đúng trục đã đo trên ViT (+6.67) nhưng trên CNN.
#   Kiểm chứng chéo  : efficientnetv2_s(IN21k) — họ CNN khác, cũng 21k
#                      -> kết luận trục pretrain có phải đặc thù ConvNeXt không.
#
# Chạy sau khi src/run_all.sh xong:
#   bash src/run_cnn2.sh                 # 3 run @224
#   NAB_448=1 bash src/run_cnn2.sh       # thêm run @448 cho model 21k tốt nhất
#
# Ghi chú optimizer: ConvNeXt/EfficientNetV2 dùng AdamW theo recipe gốc, KHÔNG
# phải SGD như resnet50/inception. Lựa chọn này đo bằng probe 4 epoch chứ không
# đoán — xem docs/02-supervised.md § "Pass 3". Hệ quả: so sánh với resnet50 ở
# trục kiến trúc có confound optimizer, nên script chạy thêm `resnet50` bằng
# AdamW làm cầu nối để bóc confound đó ra.
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/logs

. src/sweep_lock.sh
sweep_lock cnn2

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

# --- @224: ba điểm mới trên trục kiến trúc / pretrain ----------------------
run convnext_tiny_224        convnext_tiny        25 64 8 $ADAMW
run convnext_tiny_in22k_224  convnext_tiny_in22k  25 64 8 $ADAMW
run efficientnetv2_s_224     efficientnetv2_s     25 48 8 $ADAMW

# --- cầu nối bóc confound optimizer khỏi trục kiến trúc --------------------
# resnet50 đã có bản SGD (78.61). Bản AdamW cùng cờ với ConvNeXt cho phép so
# resnet50-AdamW vs convnext-AdamW, không còn lẫn hiệu ứng optimizer.
run resnet50_224_adamw       resnet50             30 64 8 $ADAMW

# --- @448 (tuỳ chọn, ~2.5-3h): độ phân giải cho CNN 21k tốt nhất -----------
# Dùng cache ảnh cạnh ngắn 512 để eval transform không phải phóng to ảnh.
if [ "${NAB_448:-0}" = "1" ]; then
  run convnext_tiny_in22k_448 convnext_tiny_in22k 25 24 8 \
      --img-size 448 --image-dir images_r512 --workers 6 $ADAMW
fi

echo "=== $(date +%H:%M:%S) CNN2 DONE"
python src/report.py --save
