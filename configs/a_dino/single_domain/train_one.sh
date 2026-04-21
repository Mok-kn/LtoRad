#!/usr/bin/env bash

CONFIG_DIR="/data1/sjc/work3/mmdetection/configs/a_dino/single_domain"
SCRIPT="/data1/sjc/work3/mmdetection/tools/train.py"

datasets=(
  "DOTA"
  "DIOR"
  "HRSSD"
)

for name in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" "$CONFIG_DIR/dino-4scale_r50_8xb2-12e_coco_${name}.py"
done