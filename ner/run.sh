#!/bin/sh

model_save_dir=model/
preds_save_dir=preds/
cache_dir=../.cache/
#load_weight=/home/user10/MatSciBERT/model/MatSciBert_concept_mlm_512_40000.pt
#load_weight=/home/user10/MatSciBERT/model/MatSciBert_random_mlm_20000.pt
#load_weight=/home/user10/MatSciBERT/model/chem_matscivec_embedding_mat2vec_masking_curriculum_random_128/MatSciBert_concept_mlm_fixed_phrase_128_100000_0.000000.pt
#load_weight=/home/user10/MatSciBERT/model/combined/chem_matscivec_embedding_mat2vec_masking_curriculum_freq_random_128/MatSciBert_concept_mlm_fixed_phrase_128_100000.pt
#load_weight=/home/user10/MatSciBERT/model/combined/chem_matscivec_embedding_mat2vec_masking_random_128/MatSciBert_concept_mlm_fixed_phrase_128_100000_15.pt

#load_weight=/home/user10/MatSciBERT/baseline/checkpoint-100000
#load_weight=/home/user10/MatSciBERT/model/checkpoint-100000
#load_weight=/home/user10/MatSciBERT/model/all/0.5/chem_matscivec_embedding_mat2vec_masking_curriculum_random_128/MatSciBert_concept_mlm_fixed_phrase_128_100000_curri_2_extra_0.200000.pt
#load_weight=/home/user10/MatSciBERT/model/all/0.5/chem_matscivec_embedding_mat2vec_masking_curriculum_random_128/MatSciBert_concept_mlm_fixed_phrase_128_100000_curri_2_extra_0.200000.pt
#load_weight=/home/user10/MatSciBERT/model/all/0.4/chem_matscivec_embedding_mat2vec_masking_curriculum_random_128/MatSciBert_concept_mlm_fixed_phrase_128_100000_curri_3_extra_0.200000_total_0.400000_raw_0.200000.pt
#load_weight=/home/user10/MatSciBERT/model/all/0.4/chem_matscivec_embedding_mat2vec_masking_curriculum_random_freq_128/MatSciBert_concept_mlm_fixed_phrase_128_100000_curri_3_extra_0.200000_total_0.300000_raw_0.150000.pt
#load_weight=/home/user10/MatSciBERT/model/all/0.4/chem_matscivec_embedding_mat2vec_masking_curriculum_random_128/MatSciBert_concept_mlm_fixed_phrase_128_100000_curri_3_extra_0.200000.pt
#load_weight=/home/user10/MatSciBERT/baseline/v3_difference_masking_mlm_128_whole_dataset_100000.pt
load_weight=/home/user10/MatSciBERT/model/matkg_128_batch_128/MatSciBert_concept_mlm_fixed_phrase_128_100000_curri_3_extra_0.000000_total_0.300000_raw_0.150000.pt

for model_name in scibert; do
    for arch in bert; do
        for dataset in sofc sofc_slot; do
            for fold in {1..5}; do
                echo $model_name $arch $dataset $fold
                CUDA_VISIBLE_DEVICES=3 python -u ner.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --architecture $arch --dataset_name $dataset --fold_num $fold --load_weight $load_weight
            done
        done

        echo $model_name $arch matscholar
        CUDA_VISIBLE_DEVICES=3 python -u ner.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --architecture $arch --dataset_name matscholar --load_weight $load_weight
        
    done
done
