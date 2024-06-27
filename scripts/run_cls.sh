#!/bin/sh

model_save_dir=./cls/model/
preds_save_dir=./cls/preds/
cache_dir=.cache/

load_weight=./MELT.pt
model_name=scibert

CUDA_VISIBLE_DEVICES=3 python -u ./cls/cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --load_weight $load_weight
