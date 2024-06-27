from mat2vec.processing import MaterialsTextProcessor
from gensim.models import Word2Vec
from tqdm import tqdm
from nltk.corpus import stopwords as nltk_stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words
import pickle
import pandas as pd
import numpy as np

import torch

device = torch.device("cuda:0")
nltk_stop_words_set = set(nltk_stop_words.words("english"))
spacy_stop_words_set = set(spacy_stop_words)
spacy_stop_words_set.add('.')
spacy_stop_words_set.add(',')
spacy_stop_words_set.add('.')
spacy_stop_words_set.add(',')
spacy_stop_words_set.add(')')
spacy_stop_words_set.add('(')
spacy_stop_words_set.add('[')
spacy_stop_words_set.add(']')
spacy_stop_words_set.add('/')
spacy_stop_words_set.add('2')
spacy_stop_words_set.add('1')
spacy_stop_words_set.add('0')
spacy_stop_words_set.add('+')
spacy_stop_words_set.add('=')
spacy_stop_words_set.add('3')
spacy_stop_words_set.add('4')
spacy_stop_words_set.add('5')
spacy_stop_words_set.add('6')
spacy_stop_words_set.add('7')
spacy_stop_words_set.add('8')
spacy_stop_words_set.add('9')


#concept_names = ["material", "property", "measurement", "form", "instrument", "device", "technique", "sample description", "application", "symmetry"]
#concept_names = ["material", "property", "measurement", "instrument", "device", "application"]
concept_names = ["material", "property", "application", "characterization_method", "descriptor", "symmetry/phase_label"]

topn = 50
score = 0.4

def process_csv(model, file_path, object_first=True):
    df = pd.read_csv(file_path)
    results = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        obj = row['Object'].lower()
        subj = row['Subject'].lower()
        if obj in model.wv.key_to_index and subj in model.wv.key_to_index:
            if object_first:
                result = model.wv[obj] - model.wv[subj]
            else:
                result = model.wv[subj] - model.wv[obj]
            results.append(result)
    print(f"Processed rows for {file_path}: {len(results)}")
    return np.mean(results, axis=0) if results else None

def get_emb(model, categories):
    
    if categories == 'material':
        file_name = 'CHM'
    elif categories == 'property':
        file_name = 'PRO'
    elif categories == 'application':
        file_name = 'APL'    
    elif categories == 'synthesis method':
        file_name = 'SYN'    
    elif categories == 'characterization_method':
        file_name = 'CMT'
    elif categories == 'descriptor':
        file_name = 'DSC'     
    elif categories == 'symmetry/phase_label':
        file_name = 'SPL'
    else:
        print("choose in right boundary")  

    embedding_1 = process_csv(model, f"./data/matkg/01.{file_name}-CHM.csv", object_first=True)
    embedding_2 = process_csv(model, f"./data/matkg/01.CHM-{file_name}.csv", object_first=False)
    final_embedding = (embedding_1 + embedding_2) / 2

    return final_embedding

def expand_vocab():
    #w2v 모델 로드
    w2v_model = Word2Vec.load("./mat2vec/data/MatSciBERT_Full/MatSciBERT_Full")
    #w2v_model = Word2Vec.load("mat2vec/training/models/pretrained_embeddings")

    #lower case for SciBERT-uncased (Can be changed in other models)
    w2v = dict()
    for key in w2v_model.wv.key_to_index:
        w2v[key.lower()] = w2v_model.wv[key]
    
    concept_vectors = dict()
    for keywords in concept_names:
        temp = get_emb(w2v_model, keywords)
        concept_vectors[keywords] = temp

    text_processor = MaterialsTextProcessor()

    concept2idx = dict()
    concept2neighbor = dict()

    #Load Extracted materia-aware entities from ChemDataExtractor
    with open("./raw_data/chem_dict.txt", "r") as f:
        x = f.readlines()

    ret = dict()

    #Save Triplet (C1, C2, R)
    num_p = 0
    f = open("./raw_data/expand_entities/triplet_chem_dict_mat2vec_matscivec_keywords_embedding_%d_similarity_%f_synthesis.txt"%(topn, score), "w")
    passed_num = 0
    for s in tqdm(x, ncols = 100):
        concept, freq = s.split("\t")

        if len(concept) < 2:
            continue
                
        if concept not in ret:
            ret[concept] = len(ret)

        if concept in w2v:
            vec = w2v[concept]

            most_similar = []

            most_temp = []
            for m in most_similar:
                if m[1] > score:
                    most_temp.append(m)
            most_similar = most_temp

            for keywords in concept_names:
                temp = concept_vectors[keywords]
                temp_similar = w2v_model.wv.most_similar([vec-temp], topn = topn)
                for m in temp_similar:
                    if m[1] > score:
                        most_similar.append(m)
                
            most_similar = set(most_similar)

        else:
            c_list = text_processor.tokenize(concept)
            if len(c_list) < 1:
                continue
            
            #Average Pooling for OOV
            vec = 0.
            for c in c_list[0]:
                if c not in w2v:
                    continue

                vec += w2v[c] / len(c_list)
            
            if isinstance(vec, float):
                passed_num += 1
                continue
            
            most_similar = []

            for keywords in concept_names:
                temp = concept_vectors[keywords]
                temp_similar = w2v_model.wv.most_similar([vec - temp], topn = topn)
                
                for m in temp_similar:
                    if m[1] > score:
                        most_similar.append(m)

            most_similar = set(most_similar)

        for c in most_similar:
            s = c[1]
            c = c[0]
            
            if c in nltk_stop_words_set:
                continue
            
            if c in spacy_stop_words_set:
                continue
            
            if len(c) < 2:
                continue
            
            #Add phrase in mat2vec
            if "_" in c:
                num_p += 1
                new_c = ""
                l = c.split("_")
                
                for ss in l:
                    new_c += ss + " "
                new_c = new_c[:-1]          
            else:
                new_c = c
            f.write("%s\t%s\t%f\n"%(concept, new_c, s))
            
            
    f.close()
    print(passed_num, num_p, len(x))

    #Triplet Dictionary
    with open('./raw_data/expand_entities/triplet_chem_dict_mat2vec_matscivec_keywords_embedding_%d_similarity_%f_synthesis.txt'%(topn, score), "r") as f:
        x = f.readlines()

    for c in tqdm(x, ncols = 50):
        c = c[:-1]
        if len(c.split('\t')) != 3:
            continue

        c1, c2, r = c.split('\t')
        
        if len(c1) == 1:
            print(c, c1, c2)
            continue
        if len(c2) == 1:
            print(c, c1, c2)
            continue
        
        if c1 not in ret:
            ret[c1] = len(ret)
        
        if c2 not in ret:
            ret[c2] = len(ret)

    print(len(ret))
    with open("./raw_data/expand_entities/extended_concept_chem_matscivec_keywords_embedding_%d_similarity_%f_synthesis.pkl"%(topn,score), "wb") as f:
        pickle.dump(ret, f)
        
expand_vocab()