#!/bin/sh

model_save_dir=model/
preds_save_dir=preds/
cache_dir=../.cache/
#load_weight=/home/user10/MatSciBERT/model/mat2vec_128_clean/MatSciBert_concept_mlm_128_whole_dataset_40000.pt
#load_weight=/home/user10/MatSciBERT/model/matkd_128_batch_128/MatSciBert_concept_mlm_128_whole_dataset_100000.pt
#load_weight=/home/user10/MatSciBERT/model/baseline/MatSciBert_concept_mlm_128_whole_dataset_100000.pt
#load_weight=/home/user10/MatSciBERT/model/chem_mat2vec_128_batch_128/MatSciBert_concept_mlm_128_whole_dataset_100000.pt
#load_weight=/home/user10/MatSciBERT/model/chem_extend_mat2vec_masking_adaptive_loss_128/MatSciBert_concept_mlm_128_100000_0.000000.pt
#load_weight=/home/user10/MatSciBERT/model/combined/chem_matscivec_embedding_mat2vec_masking_random_128/MatSciBert_concept_mlm_fixed_phrase_128_100000_15.pt
#load_weight=/home/user10/MatSciBERT/baseline/checkpoint-100000/pytorch_model.bin
#load_weight=/home/user10/MatSciBERT/baseline/DAS_100000
#load_weight=/home/user10/MatSciBERT/model/checkpoint-100000
model_name=scibert
for i in 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000; do
    #load_weight=/home/user10/MatSciBERT/baseline/checkpoint-$i
    load_weight=/home/user10/MatSciBERT/model/all/0.4/chem_matscivec_embedding_mat2vec_masking_curriculum_random_128/MatSciBert_concept_mlm_fixed_phrase_128_${i}_curri_3.pt
    echo $model_name
    #CUDA_VISIBLE_DEVICES=2 python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir
    CUDA_VISIBLE_DEVICES=3 python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --load_weight $load_weight
    

done