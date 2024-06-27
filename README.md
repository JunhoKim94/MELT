## MELT: Materials-aware Continued Pre-training for Language Model Adaptation to Materials Science

This repository is about MELT: Materials-aware Continued Pre-training for Language Model Adaptation to Materials Science submitted in EMNLP 2024. In this project, we are interested in expanding the material-aware entities for continued pre-training the PLMs for material domain adaptations.

### Requirements
 - Python 3
 - Transformers 4.6.1
 - Numpy 
 - pytorch

### Pre-processing
Prepare the pre-training corpora(e.g., scientific papers) in raw_data folder. We upload the sampled pre-training corpus in raw_data folder (```train_sampled.txt```).

1) Run ```bash scripts/bash preprocess.sh``` to normalize and split the max length(e.g., 128 length) to the raw sentences.

 - ```--train_file```: A directory containing raw text examples.
 - ```--output_train_norm_file```: A directory containing pre-processed examples.

2) Run ```bash scripts/find_entities.sh``` to preprocess the positions of material-aware entities in the pre-processed sentences.

 - ```--preprocessed_data_path```: A directory containing pre-processed examples.
 - ```--entity_path```: A directory containing material-aware entities, which are expanded by ChemDataExtractor and Mat2Vec.
 - ```--output_folder_path```: A directory containing output datasets.
 
### Pre-training
To continued pre-train PLMs, run ```bash scripts/pretrain.sh``` for distillation.

 - ```--masking_strategy```: Set the masking strategy. Choose strategies from: random, material, curriculum
 - ```--lr```: Set the learning rate.
 - ```--batch_size```: Set the batch size for conducting at once. 
 - ```--step_batch_size```: Set the batch size for updating per each step (If the memory of GPU is enough, set the batch_size and step_batch_size the same.)
 - ```--data_path```: A directory containing pre-processed examples.
 - ```--masking_ratio``` : Set the ratio for masking the material-aware entities
 - ```--curriculum_num``` : Set the number of curriculum for curriculum-aware material learning
 - ```--model_save_path```: Set the directory for saving the student model


### Fine-tuning

Run the following files with the pre-trained weights using argument name --load_weight

1) MatSciNLP

Run ```bash scripts/run_matscinlp.sh```

2) NER (SOFC-NER, SOFC-Filling, MatScholar)

Run ```bash scripts/run_ner.sh```

3) Classification (Glass Science)

Run ```bash scripts/run_cls.sh```