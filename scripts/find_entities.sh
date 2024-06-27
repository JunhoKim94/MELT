preprocessed_data_path=./raw_data/train_128_split.pkl
entity_path=./raw_data/triplet_chem_dict_mat2vec_matscivec_keywords_embedding_50_similarity_0.4.pkl
output_folder_path=./raw_data/melt


CUDA_VISIBLE_DEVICES=0 python -u ./mat2vec/chem_token_split.py --preprocessed_data_path $preprocessed_data_path --entity_path $entity_path --output_folder_path $output_folder_path