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
dictionary_dataset = "chem_matscivec_embedding"

#with open("./data/concept_dict.pkl", "rb") as f:

with open("./raw_data/triplet_chem_dict_mat2vec_matscivec_keywords_embedding_50_similarity_0.4.pkl", "rb") as f:
    concept2idx = pickle.load(f)


print(len(concept2idx))
temp = dict()

for k,v in concept2idx.items():
    if k not in temp:
        temp[k.lower()] = len(temp)
concept2idx = temp


check_dict = dict()
index = 0

#for j in range(10):
with open("./raw_data/train_128_split.pkl", "rb") as f:
    x = pickle.load(f)
    x = x["input_ids"]

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
    s = text_processor.tokenize(eg_sam, keep_sentences=False)
    #print(s)
    #print(s)
    #print(eg_sam)
    
    multi_ngram_s = split_text_ngram(s)
    #multi_ngram_s = split_text_ngram_split_blank(eg_sam)

    for token in multi_ngram_s:
        if token in nltk_stop_words_set:
            continue
        
        if token in spacy_stop_words_set:
            continue
        
        if token in concept2idx:
            concepts.append(token)
            tokenized_concepts.append(tokenizer.encode(token, add_special_tokens = False))
            
            length += 1
            
            if token not in check_dict:
                check_dict[token] = 1
            else:
                check_dict[token] += 1

    temp = x[i]
    new_concepts = []
    for j, encoded_key in enumerate(tokenized_concepts):
        cursor = 0
        length = len(encoded_key)
        for idx, encoded_id in enumerate(temp):
            
            if encoded_id == encoded_key[cursor]:
                cursor += 1
                if cursor == len(encoded_key):
                    start_point = idx + 1 - cursor
                    end_point = idx + 1
                    new_concepts.append([concepts[j], start_point, end_point])
                    cursor = 0
            else:
                cursor = 0
        
    total_concept.append(new_concepts)


print(len(train_data), len(total_concept))
ret = {"input_ids": x, "concept_positions" : total_concept}

with open('./raw_data/melt/train_data.pkl', 'wb') as f:
    pickle.dump(ret, f)

print(len(check_dict))
with open("./raw_data/melt/entity_dict_128.pkl", "wb") as f:
    pickle.dump(check_dict, f)

sorted_dict = sorted(check_dict.items(), key = lambda x: x[1], reverse = True)
#sorted_dict = ret_dict.items()

f = open("./raw_data/melt/chem_dict.txt", "w")
#for key, item in ret_dict.items():   
for key, item in sorted_dict:
    f.write("%s\t%d\n"%(key, item))
f.close()