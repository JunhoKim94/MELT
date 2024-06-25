
cache_dir=./.cache
#train_file=/scratch/maths/dual/mt6170499/matscibert/data/train_corpus.txt
#val_file=/scratch/maths/dual/mt6170499/matscibert/data/val_corpus.txt
train_norm_file=./data/train_norm.txt
model_save_dir=./model

CUDA_VISIBLE_DEVICES=0 python -u pretrain.py --train_file $train_norm_file --model_save_dir $model_save_dir --cache_dir $cache_dir --masking_ratio 0.4 --concepts "chem_matscivec_embedding" --masking_strategy "curriculum_random" --alpha 0.0 --similarity 0.4 --curriculum 3 --extra_masking 0.2 --raw_masking 0.0
