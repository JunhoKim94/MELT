
cache_dir=./.cache
model_save_dir=./model

CUDA_VISIBLE_DEVICES=0 python -u ./pretraining/pretrain.py --cache_dir $cache_dir --masking_ratio 0.4 --concepts "chem_matscivec_embedding" --masking_strategy "curriculum" --alpha 0.0 --similarity 0.4 --curriculum 3 --extra_masking 0.2 --raw_masking 0.0
