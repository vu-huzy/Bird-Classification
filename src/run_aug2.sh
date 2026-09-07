#!/usr/bin/env bash
# Pass 6b — augmentation vòng 2: CMO cho lớp hiếm + chạy lại model mạnh.
#
#   bash src/run_aug2.sh
#
# Pass 6a (`run_aug.sh`) là các ablation trên resnet50/convnext để TÌM cấu hình.
# Pass 6b này làm hai việc:
#
#   1. CMO (Park et al., CVPR 2022) — dán chim của 113 lớp hiếm (<30 ảnh train,
#      pool 2,377 ảnh) lên ảnh của lớp nhiều ảnh, dùng BBOX THẬT của repo thay vì
#      ô ngẫu nhiên như bản gốc. Quét 3 mức xác suất dán để biết ngưỡng.
#
#   2. Chạy lại các model MẠNH với cấu hình augmentation tốt nhất tìm được ở 6a,
#      MỖI model một bộ ba để quy kết được từng phần:
#         mặc định (đã có)  ->  _noerase  ->  _noerase + cmo
#      Cấu hình đó = tắt RandomErasing: pass 6a đo được nó cho +1.14 F1 lớp đuôi
#      và +1.71 recall lớp khó trên resnet50, đúng hướng tài liệu dự đoán (Cutout
#      hại FGVC vì có xác suất xoá trúng vùng phân biệt).
#
# Mọi chỉ số phải đọc bằng `python src/aug_report.py` — top-1 tổng KHÔNG đủ vì
# 113 lớp đuôi chỉ chiếm ~10% ảnh test.
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/logs

. src/sweep_lock.sh
sweep_lock aug2

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

ADAMW="--optimizer adamw --lr-backbone 1e-4 --lr-head 1e-3 --weight-decay 0.05"
BEST_AUG="--erasing-p 0"

run () {                     # run <run_name> <model> <epochs> <batch> <patience> [extra...]
  local rname=$1 model=$2 epochs=$3 bs=$4 pat=$5; shift 5
  local log="runs/logs/${rname}.log"
  if [ -f "results/${rname}/summary.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "=== $(date +%H:%M:%S) BO QUA $rname"; return 0
  fi
  echo "=== $(date +%H:%M:%S) START $rname (epochs=$epochs bs=$bs) -> $log"
  python src/train.py --model "$model" --run-name "$rname" \
    --epochs "$epochs" --batch-size "$bs" --patience "$pat" \
    --val-frac 0.15 --workers 12 "$@" > "$log" 2>&1
  local rc=$?          # phai lay NGAY: $(date) trong echo se ghi de $?
  echo "=== $(date +%H:%M:%S) DONE  $rname (exit $rc)"
  tail -n 10 "$log"
}

# --- 0. Chay lai run bi hong o pass 6a --------------------------------------
# `convnext_tiny_in22k_224_noerase` chet exit 1 vi file nguon bi sua GIUA CHUNG
# (worker spawn tren Windows import code moi nhung unpickle object cu). Khong
# phai loi thuat toan — chi can chay lai.
run convnext_tiny_in22k_224_noerase convnext_tiny_in22k 25 64 8 $ADAMW $BEST_AUG

# --- 1. CMO: quet xac suat dan ---------------------------------------------
run resnet50_224_cmo25 resnet50 30 64 8 $BEST_AUG --cmo-p 0.25
run resnet50_224_cmo50 resnet50 30 64 8 $BEST_AUG --cmo-p 0.50
run resnet50_224_cmo75 resnet50 30 64 8 $BEST_AUG --cmo-p 0.75

# --- 1b. Hai thay doi co CONG DON khong? -----------------------------------
# Pass 6a: tat erasing -> +1.14 F1 duoi / +1.71 recall kho
#          xoay 15 do  -> +1.25 F1 duoi / +2.73 recall kho  (top-1 kem hon)
# Ca hai deu giup dung nhom kho. Run nay kiem tra chung cong don hay dam nhau.
run resnet50_224_noerase_rot15 resnet50 30 64 8 $BEST_AUG --rotate 15
run resnet50_224_cmo50_rot15   resnet50 30 64 8 $BEST_AUG --rotate 15 --cmo-p 0.50

# --- 2. Chay lai model manh voi cau hinh tot nhat --------------------------
# MOI model deu chay CA HAI: `_noerase` va `_noerase + cmo`. Neu chi chay ban co
# CMO thi khong tach duoc dong gop cua CMO ra khoi dong gop cua viec tat erasing.
# Bo ba (mac dinh / noerase / noerase+cmo) cho phep quy ket tung phan.
run vit_b_16_in21k_224_noerase       vit_b_16_in21k      25 64 8 $BEST_AUG
run vit_b_16_in21k_224_cmo           vit_b_16_in21k      25 64 8 $BEST_AUG --cmo-p 0.50
run convnext_tiny_in22k_224_cmo      convnext_tiny_in22k 25 64 8 $ADAMW $BEST_AUG --cmo-p 0.50
run bioclip_224_noerase              bioclip             25 64 8 $BEST_AUG
run bioclip_224_cmo                  bioclip             25 64 8 $BEST_AUG --cmo-p 0.50

echo "=== $(date +%H:%M:%S) AUG2 DONE"
python src/report.py --save
python src/aug_report.py
