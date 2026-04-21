#!/usr/bin/env bash

CONFIG_DIR="/data1/sjc/work3/mmdetection/configs/a_deform_detr/diod"
SCRIPT="/data1/sjc/work3/mmdetection/tools/train.py"

datasets=(
  # "HR"
  # "HR_LE"
  "HR_LE_DI"
  "HR_LE_DI_DO"
)

for name in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" "$CONFIG_DIR/deformable-detr_r50_${name}.py"
done