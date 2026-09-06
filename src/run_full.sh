#!/usr/bin/env bash
# Chay TOAN BO thi nghiem trong mot lenh:
#   Pass 1 @224/299 — 6 model (cnn_scratch, resnet50, resnet101, inception_v3,
#                              vit_b_16, vit_b_16_in21k)
#   Pass 2 @448     — cnn_scratch + resnet50 (do hieu ung do phan giai)
#   Bang tong hop
#
#   bash src/stop_all.sh      # luon don sach truoc
#   bash src/run_full.sh
#
# Xem trang thai bat cu luc nao:  bash src/status.sh
set -u
cd "$(dirname "$0")/.."
. src/sweep_lock.sh
sweep_lock full

echo "############ BAT DAU $(date '+%Y-%m-%d %H:%M:%S') ############"
bash src/run_all.sh
rc1=$?
echo "############ PASS 1 (224/299) ket thuc, exit=$rc1 ############"

if [ ! -d nabirds/images_r512 ]; then
  echo "Chua co cache 512px, dang tao..."
  NAB_SHORT_SIDE=512 python src/prepare_images.py
fi

bash src/run_448.sh
rc2=$?
echo "############ PASS 2 (448) ket thuc, exit=$rc2 ############"

echo "############ BANG TONG HOP ############"
python src/report.py --save
echo "############ XONG $(date '+%Y-%m-%d %H:%M:%S') ############"
