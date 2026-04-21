#!/usr/bin/env bash

CONFIG_DIR="/data1/sjc/work3/mmdetection/configs/a_dino/diod_test"
WEIGHTS="/data1/sjc/work3/mmdetection/work_dirs/dino/diod/HR_LE_DI_DO/epoch_20.pth"
WORK_DIR_BASE="/data1/sjc/work3/mmdetection/work_dirs/dino/merged_pth/opt_test/test"
SCRIPT="/data1/sjc/work3/mmdetection/tools/test.py"

# datasets=("HRSSD" "LEVIR" "DIOR" "DOTA" "MSAR" "SSDD")
# for dataset in "${datasets[@]}"; do
#     CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" \
#         "$CONFIG_DIR/dino-4scale_r50_8xb2-12e_coco_${dataset}.py" \
#         "$WEIGHTS" \
#         --work-dir "$WORK_DIR_BASE/${dataset:0:2}"
# done

# CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" \
#     "/data1/sjc/work3/mmdetection/configs/a_dino/dino-4scale_r50_coco_Joint.py" \
#     "$WEIGHTS" \
#     --work-dir "$WORK_DIR_BASE/JO"

datasets=("HRSSD" "LEVIR" "DIOR" "DOTA")
for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" \
        "$CONFIG_DIR/dino-4scale_r50_8xb2-12e_coco_${dataset}.py" \
        "$WEIGHTS" \
        --work-dir "$WORK_DIR_BASE/${dataset:0:2}"
done