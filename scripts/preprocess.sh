train_file=./raw_data/train_sampled.txt
output_train_norm_file=./raw_data/train_normed_sampled.txt

python ./pretraining/normalize_corpus.py --train_file $train_file --output_train_norm_file $output_train_norm_file