#!/usr/bin/env bash
# Pass 4: hoàn tất hàng đợi trong PLAN.md §4 + trải rộng trục kiến trúc/quy mô.
#
#   bash src/run_more.sh
#
# Thứ tự cố ý: việc RẺ và có giá trị cao chạy trước, việc dài nhất chạy cuối, và
# ensemble chốt hạ vì nó cần checkpoint của mọi run đứng trước.
#
# | # | run | trả lời câu gì |
# |---|-----|----------------|
# | 1 | bioclip              | điểm thứ TƯ trên trục pretrain: không → IN1k → IN21k → ToL-10M.
# |   |                      | CÙNG ViT-B/16, CÙNG SGD, CÙNG lr với vit_b_16_in21k ⇒ chênh lệch
# |   |                      | quy hết về NGUỒN pretrain. Đây là thí nghiệm ImageNet→iNaturalist
# |   |                      | của Cui et al. CVPR'18 làm lại với foundation model sinh học 2024.
# | 2 | mobilenet_v3_large   | 5.5M param có pretrain vs cnn_scratch 5.0M KHÔNG pretrain (60.72):
# |   |                      | tách đòn bẩy pretrain ở CÙNG mức dung lượng model.
# | 3 | densenet121          | dense connectivity (2017), 8M param — CNN nhỏ mà sâu.
# | 4 | resnext50_32x4d      | grouped conv vs plain ResNet-50, gần như cùng số tham số.
# | 5 | swin_tiny_in22k      | 27.9M vs convnext_tiny_in22k 28.2M, cùng IN22k, cùng 224
# |   |                      | ⇒ CNN vs transformer phân cấp ở cùng cỡ và cùng nguồn pretrain.
# | 6 | vgg16_bn             | kiến trúc 2014, không skip connection — mốc dưới của trục thời gian.
# | 7 | vit_b_16_in21k @448  | C4 trong PLAN: README đang SUY ĐOÁN "khoảng cách với TransFG chủ
# |   |                      | yếu là độ phân giải" mà chưa hề đo. Đây là phép đo đó.
# | 8 | ensemble + TTA       | hai đòn bẩy miễn phí, không train gì thêm.
#
# Optimizer: SGD cho CNN BatchNorm và cho ViT (đã chứng minh chạy tốt ở pass 1);
# AdamW cho Swin vì đó là recipe gốc của nó và vì bài học pass 3 (lr SGD của ViT
# bê sang kiến trúc phân cấp có LayerNorm thì tụt 49 điểm).
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/logs

. src/sweep_lock.sh
sweep_lock more

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

# 1. Trục pretrain, điểm thứ tư — recipe PHẢI giống hệt vit_b_16_in21k_224.
run bioclip_224            bioclip            25 64 8

# 2-4, 6. CNN cổ điển / hiệu quả, cùng IN1k @224, SGD như resnet50.
run mobilenet_v3_large_224 mobilenet_v3_large 30 64 8
run densenet121_224        densenet121        30 64 8
run resnext50_32x4d_224    resnext50_32x4d    30 64 8
run vgg16_bn_224           vgg16_bn           30 32 8

# 5. Transformer phân cấp cùng cỡ ConvNeXt-T, cùng IN22k.
run swin_tiny_in22k_224    swin_tiny_in22k    25 64 8 $ADAMW

# 7. C4 — độ phân giải cho ViT. Grad checkpointing bắt buộc: 784 token/ảnh.
run vit_b_16_in21k_448     vit_b_16_in21k     25 16 8 \
    --img-size 448 --image-dir images_r512 --workers 6 --grad-checkpointing

# 8. Ensemble + TTA — cần mọi checkpoint ở trên nên phải chạy cuối.
echo "=== $(date +%H:%M:%S) START ensemble + TTA"
python src/ensemble.py --cache --report > runs/logs/ensemble.log 2>&1
rc=$?                # phai lay NGAY: $(date) trong echo se ghi de $?
echo "=== $(date +%H:%M:%S) DONE  ensemble (exit $rc)"
tail -n 30 runs/logs/ensemble.log

echo "=== $(date +%H:%M:%S) MORE DONE"
python src/report.py --save
