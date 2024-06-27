
cache_dir=./.cache
model_save_dir=./model
data_path=./raw_data/melt/train_data.pkl

CUDA_VISIBLE_DEVICES=0 python -u ./pretraining/pretrain.py --cache_dir $cache_dir --masking_ratio 0.4 --concepts "chem_matscivec_embedding" --masking_strategy "curriculum" --curriculum 3 --data_path $data_path
