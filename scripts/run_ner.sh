#!/bin/sh

model_save_dir=./ner/model/
preds_save_dir=./ner/preds/
cache_dir=.cache/

load_weight=./MELT.pt

for model_name in scibert; do
    for arch in bert; do
        for dataset in sofc sofc_slot; do
            for fold in {1..5}; do
                echo $model_name $arch $dataset $fold
                CUDA_VISIBLE_DEVICES=0 python -u ner.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --architecture $arch --dataset_name $dataset --fold_num $fold --load_weight $load_weight
            done
        done

        echo $model_name $arch matscholar
        CUDA_VISIBLE_DEVICES=0 python -u ner.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --architecture $arch --dataset_name matscholar --load_weight $load_weight
        
    done
done
