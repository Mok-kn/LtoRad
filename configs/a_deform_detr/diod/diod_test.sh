#!/usr/bin/env bash

CONFIG_DIR="/data1/sjc/work3/mmdetection/configs/a_deform_detr/diod"
WEIGHTS="/data1/sjc/work3/mmdetection/work_dirs/deform_detr/diod/HR_LE_DI_DO/epoch_20.pth"
WORK_DIR_BASE="/data1/sjc/work3/mmdetection/work_dirs/deform_detr/diod/test"
SCRIPT="/data1/sjc/work3/mmdetection/tools/test.py"

datasets=("HR" "HR_LE" "HR_LE_DI" "HR_LE_DI_DO")
for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" \
        "$CONFIG_DIR/deformable-detr_r50_${dataset}.py" \
        "$WEIGHTS" \
        --work-dir "$WORK_DIR_BASE/${dataset}"
done

# CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" \
#     "/data1/sjc/work3/mmdetection/configs/a_dino/dino-4scale_r50_coco_Joint.py" \
#     "$WEIGHTS" \
#     --work-dir "$WORK_DIR_BASE/JO"