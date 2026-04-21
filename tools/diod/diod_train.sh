#!/usr/bin/env bash

CONFIG_DIR="/data1/sjc/work3/mmdetection/configs/a_dino/diod_freeze_backbone"
SCRIPT="/data1/sjc/work3/mmdetection/tools/train.py"

datasets=(
  "HR_LE"
  "HR_LE_DI"
  "HR_LE_DI_DO"
  "HR_LE_DI_DO_MS"
  "HR_LE_DI_DO_MS_SS"
)

for name in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" "$CONFIG_DIR/dino-4scale_r50_coco_${name}.py"
done
