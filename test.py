import pickle
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

concept_name = "chem"

with open("/home/user10/MatSciBERT/chem_dict.txt", "r") as f:
    x = f.readlines()
    origin_dict = dict()
    for s in x:
        c, freq = s[:-1].split("\t")
        
        if c not in origin_dict:
            origin_dict[c] = len(origin_dict)


print("test!!")

ret_dict = dict()
s_length = 0
avg_concept_length = 0.
total_length = 0.
for i in range(10):
    #with open("/home/user10/MatSciBERT/data/all/concept_128_ngram_%s/split_real/train_data_%d.pkl"%(concept_name, i+1), "rb") as f:
    with open("/home/user10/MatSciBERT/data/concept_128_ngram_%s/split_real/train_data_%d.pkl"%(concept_name, i+1), "rb") as f:
    
        x = pickle.load(f)
        
    temp = dict()
    temp["input_ids"] = x["train_data"]
    temp["concept_positions"] = x["concepts"]
    x = temp
    
    sentences = x["input_ids"]
    concepts = x['concept_positions']
    s_length += len(concepts)
    for sent, cs in tqdm(zip(sentences, concepts), ncols = 100):
        total_length += len(cs)
        if len(cs) < 1:
            continue
        
        for s in cs:        
            avg_concept_length += s[2] - s[1]
            con = s[0]
            
            if con in ret_dict:
                ret_dict[con] += 1                
            else:
                ret_dict[con] = 1

print(total_length, s_length, avg_concept_length / s_length)
#print(ret_dict, len(ret_dict))
print(len(ret_dict), total_length / len(ret_dict))
sorted_dict = sorted(ret_dict.items(), key = lambda x: x[1], reverse = True)
#sorted_dict = ret_dict.items()

f = open("/home/user10/MatSciBERT/data/concept_128_ngram_%s/chem_dict.txt"%concept_name, "w")
#for key, item in ret_dict.items():   
for key, item in sorted_dict:
    f.write("%s\t%d\n"%(key, item))
f.close()
