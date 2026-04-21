#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python /data1/sjc/work3/mmdetection/tools/train.py /data1/sjc/work3/mmdetection/configs/a_dino/diod_freeze_encoder_decoder_mix/dino-4scale_r50_coco_HR_LE_DI_DO_MS.py
CUDA_VISIBLE_DEVICES=0 python /data1/sjc/work3/mmdetection/tools/train.py /data1/sjc/work3/mmdetection/configs/a_dino/diod_freeze_encoder_decoder_mix/dino-4scale_r50_coco_HR_LE_DI_DO_MS_SS.py

