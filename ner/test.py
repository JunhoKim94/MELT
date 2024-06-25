import os
import pickle
import sys
import multiprocessing as mp
sys.path.append('../')

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from tqdm import tqdm
from chemdataextractor.doc import Paragraph

import torch
from torch import nn

import ner_datasets
from models import BERT_CRF
from normalize_text import normalize

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

import chemdataextractor as cde

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

def apply_parallel(func_name, l):
    p = min(len(l), mp.cpu_count())
    with mp.Pool(processes=p) as pool:
        r = list(tqdm(pool.imap(func_name, l), total=len(l)))
    return r

def tokenize_caption(c):
    para = Paragraph(normalize(c))
    ret = []
    for sent in para.tokens:
        ret.append([t.text for t in sent])
    return ret


root_dir = '../'
cache_dir = os.path.join(root_dir, '.cache')
output_dir = os.path.join(root_dir, 'ner/output_matscibert_matscholar')
model_name = os.path.join(root_dir, 'ner/models/matscholar')
to_normalize = True


# captions is the list of sentences from which entities need to be extracted

captions= ['Glasses are emerging as promising and efficient solid electrolytes for all-solid-state sodium-ion batteries.',
           'The current study shows a significant enhancement in crack resistance (from 11.3 N to 32.9 N) for Na3Al1.8Si1.65P1.8O12 glass (Ag-0 glass) upon Na+-Ag+ ion-exchange (IE) due to compressive stresses generated in the glass surface while the ionic conductivity values (∼10−5 S/cm at 473 K) were retained. ',
           'In this study, magic angle spinning-nuclear magnetic resonance (MAS-NMR), molecular dynamics (MD) simulations, Vickers micro hardness, and impedance spectroscopic techniques were used to evaluate the intermediate-range structure, atomic structure, crack resistance and conductivity of the glass.',
           'Selected beam geometry allows us to suppress the bulk contribution to sum-frequency generation from crystalline quartz and use sum-frequency vibrational spectroscopy to study water/α-quartz interfaces with different bulk pH values.',
           'XRD patterns of glass-ceramics sintered at different holding times; identifying rutile TiO2 crystal grains.']
           
# del df
captions = [c for c in captions if type(c) == str]

#tok_captions = apply_parallel(tokenize_caption, captions)

tok_captions = tokenize_caption(captions[1])

#print(tok_captions)

sum_tok_captions = []
for t in tok_captions:
    sum_tok_captions += t
tok_captions = sum_tok_captions

train_X, train_y = ner_datasets.get_ner_data('matscholar', norm=to_normalize)[:2]
print(len(train_X))

tokenizer_kwargs = {
    'cache_dir': cache_dir,
    'use_fast': True,
    'revision': 'main',
    'use_auth_token': None,
    'model_max_length': 512
}
tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert', **tokenizer_kwargs)
a = tokenizer.tokenize(captions[1])
print(a, tok_captions)

from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')
model = AutoModelForMaskedLM.from_pretrained('m3rg-iitd/matscibert')

print(model)