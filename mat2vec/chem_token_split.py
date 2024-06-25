from mat2vec.processing import MaterialsTextProcessor
from chemdataextractor.doc import Paragraph
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
    AdamW,
)
import pickle
import nltk
nltk.download('stopwords')

from tqdm import tqdm
from nltk.corpus import stopwords as nltk_stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words

def split_text_ngram(text):
    #word_list = text.split()
    word_list = text
    word_list_len = len(word_list)
    multigram_cpt_list = []
    bigram_cpt_list = []
    trigram_cpt_list = []
    fourgram_cpt_list = []
    fivegram_cpt_list = []

    multigram_cpt_list += word_list

    for bigram_idx in range(word_list_len - 1):
        bi_list = word_list[bigram_idx: 2 + bigram_idx]
        bigram_cpt = bi_list[0] + ' ' + bi_list[1]
        bigram_cpt_list.append(bigram_cpt)
        multigram_cpt_list.append(bigram_cpt)

    for trigram_idx in range(word_list_len - 2):
        tri_list = word_list[trigram_idx: 3 + trigram_idx]
        trigram_cpt = tri_list[0] + ' ' + tri_list[1] + ' ' + tri_list[2]
        trigram_cpt_list.append(trigram_cpt)
        multigram_cpt_list.append(trigram_cpt)

    for fourgram_idx in range(word_list_len - 3):
        four_list = word_list[fourgram_idx: 4 + fourgram_idx]
        fourgram_cpt = four_list[0] + ' ' + four_list[1] + ' ' + four_list[2] + ' ' + four_list[3]
        fourgram_cpt_list.append(fourgram_cpt)
        multigram_cpt_list.append(fourgram_cpt)

    for fivegram_idx in range(word_list_len - 4):
        five_list = word_list[fivegram_idx: 5 + fivegram_idx]
        fivegram_cpt = five_list[0] + ' ' + five_list[1] + ' ' + five_list[2] + ' ' + five_list[3] + ' ' + five_list[4]
        fivegram_cpt_list.append(fivegram_cpt)
        multigram_cpt_list.append(fivegram_cpt)

    multigram_cpt_list.extend(word_list)
    return multigram_cpt_list

def split_text_ngram_split_blank(text):
    word_list = text.split()
    #word_list = text
    word_list_len = len(word_list)
    multigram_cpt_list = []
    bigram_cpt_list = []
    trigram_cpt_list = []
    fourgram_cpt_list = []
    fivegram_cpt_list = []

    multigram_cpt_list += word_list

    for bigram_idx in range(word_list_len - 1):
        bi_list = word_list[bigram_idx: 2 + bigram_idx]
        bigram_cpt = bi_list[0] + ' ' + bi_list[1]
        bigram_cpt_list.append(bigram_cpt)
        multigram_cpt_list.append(bigram_cpt)

    for trigram_idx in range(word_list_len - 2):
        tri_list = word_list[trigram_idx: 3 + trigram_idx]
        trigram_cpt = tri_list[0] + ' ' + tri_list[1] + ' ' + tri_list[2]
        trigram_cpt_list.append(trigram_cpt)
        multigram_cpt_list.append(trigram_cpt)

    for fourgram_idx in range(word_list_len - 3):
        four_list = word_list[fourgram_idx: 4 + fourgram_idx]
        fourgram_cpt = four_list[0] + ' ' + four_list[1] + ' ' + four_list[2] + ' ' + four_list[3]
        fourgram_cpt_list.append(fourgram_cpt)
        multigram_cpt_list.append(fourgram_cpt)

    for fivegram_idx in range(word_list_len - 4):
        five_list = word_list[fivegram_idx: 5 + fivegram_idx]
        fivegram_cpt = five_list[0] + ' ' + five_list[1] + ' ' + five_list[2] + ' ' + five_list[3] + ' ' + five_list[4]
        fivegram_cpt_list.append(fivegram_cpt)
        multigram_cpt_list.append(fivegram_cpt)

    multigram_cpt_list.extend(word_list)
    return multigram_cpt_list

nltk_stop_words_set = set(nltk_stop_words.words("english"))
spacy_stop_words_set = set(spacy_stop_words)
spacy_stop_words_set.add('.')
spacy_stop_words_set.add(',')
spacy_stop_words_set.add(')')
spacy_stop_words_set.add('(')
spacy_stop_words_set.add('[')
spacy_stop_words_set.add(']')
spacy_stop_words_set.add('/')
spacy_stop_words_set.add('2')
spacy_stop_words_set.add('1')
spacy_stop_words_set.add('+')
spacy_stop_words_set.add('=')
spacy_stop_words_set.add('3')
spacy_stop_words_set.add('4')
spacy_stop_words_set.add('5')
spacy_stop_words_set.add('6')
spacy_stop_words_set.add('7')
spacy_stop_words_set.add('8')
spacy_stop_words_set.add('9')
spacy_stop_words_set.add('10')

tokenizer_kwargs = {
    'use_fast': True,
    'revision': 'main',
    'use_auth_token': None,
    'model_max_length': 512
}

text_processor = MaterialsTextProcessor()
tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert', **tokenizer_kwargs)

#print(sentences)
test_mode = False
max_length = 126
#dictionary_dataset = "chem_matscivec_embedding"
#dictionary_dataset = "combine"
dictionary_dataset = "matkg"
#dictionary_dataset = "matkg_extend"
#dictionary_dataset = "chem_matscivec"
#dictionary_dataset = "mat_clean"
#dictionary_dataset = "chem_matscivec_embedding"
#dictionary_dataset = "chem_matscivec"

#with open("./data/concept_dict.pkl", "rb") as f:

if dictionary_dataset == "chem_extend":
    #Chem+Mat2Vec
    with open("/home/user10/MatSciBERT/mat2vec/data/extended_concept_chem.pkl", "rb") as f:
        concept2idx = pickle.load(f)
elif dictionary_dataset == "matkg":
    #MatKG
    with open("/home/user10/MatSciBERT/matkg/chem_dict.pkl", "rb") as f:
        concept2idx = pickle.load(f)
    
elif dictionary_dataset == "matkg_extend":
    with open("/home/user10/MatSciBERT/mat2vec/data/extended_concept_matkg.pkl", "rb") as f:
        concept2idx = pickle.load(f)
    
elif dictionary_dataset == "combine":
    with open("/home/user10/MatSciBERT/mat2vec/data/extended_concept_chem.pkl", "rb") as f:
        concept2idx1 = pickle.load(f)
    with open("/home/user10/MatSciBERT/matkg/chem_dict.pkl", "rb") as f:
        concept2idx2 = pickle.load(f)
    
    concept2idx = dict()
    for k,v in concept2idx1.items():
        if k not in concept2idx:
            concept2idx[k] = len(concept2idx)
    
    for k,v in concept2idx2.items():
        if k not in concept2idx:
            concept2idx[k] = len(concept2idx)
    
elif dictionary_dataset == "chem_clean":
    with open("/home/user10/MatSciBERT/mat2vec/data/cleaned/02.extended_concept_chem_sort.txt", "r") as f:
        x = f.readlines()    
    concept2idx = dict()
    for s in x:
        concept2idx[s[:-1]] = len(concept2idx)

elif dictionary_dataset == "mat_clean":
    with open("/home/user10/MatSciBERT/mat2vec/data/cleaned/02.extended_concept_MatKG_sort.txt", "r") as f:
        x = f.readlines()
    concept2idx = dict()
    for s in x:
        concept2idx[s[:-1]] = len(concept2idx)

elif dictionary_dataset == "chem_matscivec":
    with open("/home/user10/MatSciBERT/mat2vec/data/mat2vec_ours/extended_concept_chem_matscivec.pkl", "rb") as f:
        concept2idx = pickle.load(f)

elif dictionary_dataset == "chem_matscivec_embedding":
    with open("/home/user10/MatSciBERT/mat2vec/data/mat2vec_ours/extended_concept_chem_matscivec_keywords_embedding_50.pkl", "rb") as f:
        concept2idx = pickle.load(f)


print(len(concept2idx))
temp = dict()

for k,v in concept2idx.items():
    if k not in temp:
        temp[k.lower()] = len(temp)
concept2idx = temp

'''
num_p = 0
for k, v in concept2idx.items():
    if "_" in k:
        num_p += 1
        t = ""
        for s in k.split("_"):
            t += s + " "
        k = k[:-1]

    if k not in temp:
        temp[k.lower()] = len(temp)

concept2idx = temp
print(len(concept2idx), num_p)
'''


#with open("/home/user10/MatSciBERT/data/train_norm.txt", "r") as f:
#    x = f.readlines()
    #sents = f.read().strip().split('\n')

#with open("/home/user10/MatSciBERT/mat2vec/data/quick_preprocessed_512_extend.pkl", "rb") as f:
#    x = pickle.load(f)

#x = x["train_data"]
#print(x[:10], len(x[0]))
#exit()
#with open("/home/user10/MatSciBERT/mat2vec/data/temp.pkl", "wb") as f:
#    pickle.dump(x[:10000], f)
'''
with open("/home/user10/MatSciBERT/data/concept_126_mat2vec_clean_real/mat2vec_concept_preprocessed_126.pkl", "rb") as f:
    x = pickle.load(f)

x = x["train_data"]

import random

random.shuffle(x)

total = len(x)
folder = total // 10
for i in range(10):
    with open("/home/user10/MatSciBERT/data/concept_126_mat2vec_clean_real/split/train_data_%d"%(i+1), "wb") as f:
        temp = x[i * folder : (i+1) * folder]
        pickle.dump(temp, f)
'''
#sentences = ''
#for s in x:
#    sentences += s


check_dict = dict()
index = 0

for j in range(10):
    with open("/home/user10/MatSciBERT/data/concept_128_ngram_chem/split/train_data_%d.pkl"%(j+1), "rb") as f:
        x = pickle.load(f)
        x = x["train_data"]

    train_data = []
    total_concept = []
    total_tokenized_concept = []
    #concepts = []
    length = 0
    temp = []

    for i in tqdm(range(len(x)), ncols = 100):
        concepts = []
        tokenized_concepts = []
        eg_sam = x[i]
        #print(text_processor.tokenize(eg_sam))
        #print(tokenizer.tokenize(eg_sam))

        eg_sam = tokenizer.decode(eg_sam, add_special_tokens = False)
        #temp = []
        #s = text_processor.tokenize(eg_sam, keep_sentences=False)
        #print(s)
        #print(s)
        #print(eg_sam)
        
        #multi_ngram_s = split_text_ngram(s)
        multi_ngram_s = split_text_ngram_split_blank(eg_sam)

        for token in multi_ngram_s:
            #tokenized = tokenizer.tokenize(token, add_special_tokens = False)
            #curr_len = len(temp)
            #tokenized_len = len(tokenized)
            #temp += [token]
            #temp += tokenized
            
            if token in nltk_stop_words_set:
                continue
            
            if token in spacy_stop_words_set:
                continue
            
            if token in concept2idx:
                #print(token)
                #concepts.append((token, curr_len, curr_len + tokenized_len))
                concepts.append(token)
                tokenized_concepts.append(tokenizer.encode(token, add_special_tokens = False))
                
                length += 1
                
                if token not in check_dict:
                    check_dict[token] = len(check_dict)
            

        temp = x[i]
        new_concepts = []
        for j, encoded_key in enumerate(tokenized_concepts):
            cursor = 0
            length = len(encoded_key)
            #print(encoded_key)
            for idx, encoded_id in enumerate(temp):
                
                
                if encoded_id == encoded_key[cursor]:
                    cursor += 1
                    if cursor == len(encoded_key):
                        start_point = idx + 1 - cursor
                        end_point = idx + 1
                        #start_n_end_point_list.append([start_point, end_point])
                        #concepts[j] = [concepts[j], start_point, end_point]
                        new_concepts.append([concepts[j], start_point, end_point])
                        
                        #print(concepts[j], tokenizer.decode(temp[start_point:end_point], add_special_tokens = False))
                        cursor = 0
                else:
                    cursor = 0
            
        total_concept.append(new_concepts)
        #print(new_concepts)
        #total_tokenized_concept.append(tokenized_concepts)   
        #temp += tokenizer.encode(token, add_special_tokens = False)
            
        #print(len(temp), length)

    print(len(train_data), len(total_concept))
    ret = {"train_data": x, "concepts" : total_concept}

    with open('/home/user10/MatSciBERT/data/concept_128_ngram_%s/split_real/train_data_%d.pkl'%(dictionary_dataset, index+1), 'wb') as f:
        pickle.dump(ret, f)
        
    index += 1
    
print(len(check_dict))
with open("/home/user10/MatSciBERT/data/concept_128_ngram_%s/check_dict_128_real.pkl"%dictionary_dataset, "wb") as f:
    pickle.dump(check_dict, f)
    