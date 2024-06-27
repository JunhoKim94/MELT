import random
from tqdm import tqdm
from argparse import ArgumentParser
from tokenizers.normalizers import BertNormalizer
from transformers import AutoTokenizer
import pickle

parser = ArgumentParser()
parser.add_argument('--train_file', required=True, type=str)
parser.add_argument('--output_train_norm_file', required=True, type=str)
args = parser.parse_args()


f = open('./vocab_mappings.txt')
mappings = f.read().strip().split('\n')
f.close()

mappings = {m[0]: m[2:] for m in mappings}

norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)

def normalize_and_save(file_path, save_file_path):
    f = open(file_path)
    corpus = f.read().strip().split('\n')
    f.close()
    
    random.seed(42)
    corpus = [norm.normalize_str(sent) for sent in tqdm(corpus)]
    corpus_norm = []
    for sent in tqdm(corpus):
        norm_sent = ""
        for c in sent:
            if c in mappings:
                norm_sent += mappings[c]
            elif random.uniform(0, 1) < 0.3:
                norm_sent += c
            else:
                norm_sent += ' '
        corpus_norm.append(norm_sent)
    
    f = open(save_file_path, 'w')
    f.write('\n'.join(corpus_norm))
    f.close()


def full_sent_tokenize(file_name):
    
    tokenizer_kwargs = {
    'use_fast': True,
    'use_auth_token': None,
    }
    tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert', **tokenizer_kwargs)
    max_seq_length = 128
    start_tok = tokenizer.convert_tokens_to_ids('[CLS]')
    sep_tok = tokenizer.convert_tokens_to_ids('[SEP]')
    #pad_tok = tokenizer.convert_tokens_to_ids('[PAD]')

    f = open(file_name, 'r')
    sents = f.read().strip().split('\n')
    f.close()
    
    tok_sents = [tokenizer(s, padding=False, truncation=False)['input_ids'] for s in tqdm(sents)]
    for s in tok_sents:
        s.pop(0)
    
    res = [[]]
    l_curr = 0
    
    for s in tqdm(tok_sents, ncols = 100):
        l_s = len(s)
        idx = 0
        while idx < l_s - 1:
            if l_curr == 0:
                res[-1].append(start_tok)
                l_curr = 1
            s_end = min(l_s, idx + max_seq_length - l_curr) - 1
            res[-1].extend(s[idx:s_end] + [sep_tok])
            idx = s_end
            if len(res[-1]) == max_seq_length:
                res.append([])
            l_curr = len(res[-1])
    
    for s in res[:-1]:
        assert s[0] == start_tok and s[-1] == sep_tok
        assert len(s) == max_seq_length
        
    attention_mask = []
    for s in res:
        attention_mask.append([1] * len(s) + [0] * (max_seq_length - len(s)))
    
    return {'input_ids': res, 'attention_mask': attention_mask}


normalize_and_save(args.train_file, args.output_train_norm_file)
train_corpus = full_sent_tokenize(args.output_train_norm_file)
with open('./raw_data/train_128_split.pkl', "wb") as f:
    pickle.dump(train_corpus, f)