import pickle
from tqdm import tqdm
import numpy as np
import copy

similarity = 0.4

with open("./raw_data/expand_entities/triplet_chem_dict_mat2vec_matscivec_keywords_embedding_50_similarity_0.400000_all.txt", "r") as f:
    x = f.readlines()

ret_dict = dict()
for s in tqdm(x, ncols = 100):
    c1, c2, r = s[:-1].split("\t")
    #c1, c2 = s[:-1].split("\t")
    
    r = float(r)

    if r < similarity:
        continue

    c1 = c1.lower()
    c2 = c2.lower()

    if c1 not in ret_dict:
        ret_dict[c1] = set([c2])
    else:
        ret_dict[c1].add(c2)
    
    if c2 not in ret_dict:
        ret_dict[c2] = set([c1])
    else:
        ret_dict[c2].add(c1)
        
length = 0
length_list = []
freq_list = []

skipped_length = 0
trivial = 0
non_de = 0

key2length = dict()

for k in ret_dict.keys():
    
    temp = ret_dict[k]
    length += len(temp)   
    
    length_list.append(len(temp))
    key2length[k] = len(temp)
    #freq_list.append(c2f[k])


print(len(key2length))
#print(skipped_length, trivial, non_de)
sorted_dict = sorted(key2length.items(), key = lambda x: x[1], reverse = True)


with open("./raw_data/melt/chem_dict.txt", "r") as f:
    x = f.readlines()

corpus2freq = dict()
for s in x:
    concept, freq = s[:-1].split("\t")
    corpus2freq[concept] = freq

rel2freq = []
idx = 0
corpus2rel = dict()
for k in corpus2freq.keys():
    if k not in key2length:
        idx += 1
        print(k)
        continue
    rel2freq.append([key2length[k], corpus2freq[k]])
    corpus2rel[k] = key2length[k]
    
x = []
y = []
for s in rel2freq:
    #if s[0] < 100:
    x.append(s[0])
    y.append(s[1])

print(idx)

sorted_corpus = sorted(corpus2rel.items(), key = lambda x: x[1], reverse = True)
total_length = len(sorted_corpus)

for num_curriculum in [2, 3, 4]:

    t = dict()
    ret = []
    for i in range(num_curriculum):
        l = sorted_corpus[i * total_length // num_curriculum : (i+1) * (total_length // num_curriculum)]
        for n in l:
            t[n[0]] = n[1]
        new_temp = copy.deepcopy(t)
        ret.append(new_temp)
        print(len(new_temp))

    with open("./raw_data/melt/curriculum_%d.pkl"%num_curriculum, "wb") as f:
        pickle.dump(ret, f)

    t = dict()
    ret = []
    for i in range(num_curriculum - 1, -1, -1):
        l = sorted_corpus[i * total_length // num_curriculum : (i+1) * (total_length // num_curriculum)]
        for n in l:
            t[n[0]] = n[1]
        new_temp = copy.deepcopy(t)
        ret.append(new_temp)
        print(len(new_temp))

    with open("./raw_data/melt/curriculum_anti_%d.pkl"%num_curriculum, "wb") as f:
        pickle.dump(ret, f)


    '''
    t = dict()
    ret = []
    num_of_concept = 9999
    limit_freq = 100

    num_continued = 0
    base_concept = dict()
    for s in sorted_corpus:    
        if int(corpus2freq[s[0]]) < limit_freq:
            num_continued += 1
            continue
        
        t[s[0]] = len(t)
        
        if len(t) > num_of_concept:
            break

    print(len(t), num_continued)
    ret.append(copy.deepcopy(t))

    new_dict = copy.deepcopy(t)
    for i in range(num_curriculum):


        for k in new_dict.keys():
            #print(len(ret_dict[k]))
            for c in ret_dict[k]:
                if c not in t:
                    t[c] = len(t)            

        ret.append(copy.deepcopy(t))
        print(len(t))
        new_dict = copy.deepcopy(t)

    with open("/mnt/user10/bert/concept_128_ngram_%s/curriculum_concept_%d.pkl"%(concept_name, num_curriculum), "wb") as f:
        pickle.dump(ret, f)
    '''