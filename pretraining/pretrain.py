import os
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import torch
from transformers import (
    AutoConfig,
    BertForMaskedLM,
    AutoTokenizer,
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorForLanguageModeling
)
from transformers import get_linear_schedule_with_warmup
import pickle
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import copy
import random
from masking_strategy import DataCollatorForWholeWordMask_custom

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path

def full_sent_tokenize(file_name):
    f = open(file_name, 'r')
    sents = f.read().strip().split('\n')
    f.close()
    
    tok_sents = [tokenizer(s, padding=False, truncation=False)['input_ids'] for s in tqdm(sents, ncols = 100)]
    for s in tok_sents:
        s.pop(0)
    
    res = [[]]
    l_curr = 0
    
    for s in tok_sents:
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

class MSC_Dataset(torch.utils.data.Dataset):
    def __init__(self, inp):
        self.inp = inp

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.inp.items()}
        return item

    def __len__(self):
        return len(self.inp['input_ids'])

parser = ArgumentParser()
parser.add_argument('--train_file', required=True, type=str)
parser.add_argument('--val_file', required=True, type=str)
parser.add_argument('--model_save_dir', required=True, type=str)
parser.add_argument('--cache_dir', default="/home/user10/MatSciBERT/.cache", type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--weight_decay', default=1e-2, type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--step_batch_size', default = 128, type = int)
parser.add_argument('--masking_ratio', default=0.4, type=float)
parser.add_argument('--concepts', default="chem", type = str)
parser.add_argument('--masking_strategy', default="random", type = str)
parser.add_argument('--alpha', default=-0.3, type = float)

parser.add_argument('--curriculum_num', default=3, type = int)
parser.add_argument('--similarity', default=0.4, type = str)
parser.add_argument('--extra_masking', default=0.2, type = float)
parser.add_argument('--raw_masking', default=0.15, type = float)


args = parser.parse_args()

model_revision = 'main'
model_name = 'allenai/scibert_scivocab_uncased'
#model_name = "bert-base-uncased"
cache_dir = ensure_dir(args.cache_dir) if args.cache_dir else None
output_dir = ensure_dir(args.model_save_dir)

writer = SummaryWriter(comment=f" || MatSciBERT_concept_masking_128 || {args.batch_size}_{args.lr}_{args.concepts}_{args.alpha}_{args.masking_strategy}_{args.masking_ratio}")

assert os.path.exists(args.train_file)
assert os.path.exists(args.val_file)

SEED = random.randint(0, 10000)
#SEED = 42
set_seed(SEED)

config_kwargs = {
    'cache_dir': cache_dir,
    'revision': model_revision,
    'use_auth_token': None,
}
config = AutoConfig.from_pretrained(model_name, **config_kwargs)

tokenizer_kwargs = {
    'cache_dir': cache_dir,
    'use_fast': True,
    'revision': model_revision,
    'use_auth_token': None,
}

tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
#max_seq_length = 512
max_seq_length = 128

start_tok = tokenizer.convert_tokens_to_ids('[CLS]')
sep_tok = tokenizer.convert_tokens_to_ids('[SEP]')
pad_tok = tokenizer.convert_tokens_to_ids('[PAD]')
mask_tok = tokenizer.convert_tokens_to_ids('[MASK]')

class MSC_Dataset_for_concept(torch.utils.data.Dataset):
    def __init__(self, inp):
        self.inp = inp
        
        #self.inp["input_ids"] = torch.Tensor(self.inp["input_ids"])
        

    def __getitem__(self, idx):
        
        input_ids = self.inp['input_ids'][idx]
        
        if len(input_ids) > 128:
            input_ids = input_ids[:128]
        
        input_ids = [start_tok] + input_ids + [sep_tok]
        #concept_position = [[p[0] + 1, p[1] + 1] for p in self.inp['concept_positions'][idx]]
        concept_position = [[p[1] + 1, p[2] + 1, p[0]] for p in self.inp['concept_positions'][idx]]
        
        
        item = {"input_ids" : input_ids, 'concept_positions' : concept_position}
        #item = {key: torch.tensor(val[idx]) for key, val in self.inp.items()}
        return item

    def __len__(self):
        #return len(self.inp['input_ids'])
        return len(self.inp['input_ids'])


def concept_masking(origin_input_ids, input_ids, labels, concept_positions, masking_ratio = 0.4, masked_concept_freq = None):
    batch_size = input_ids.size(0)
    
    masked_concept_token = 0
    
    concept_masked_position = torch.zeros_like(labels).fill_(-100).cuda()
    
    #print(concept_positions)
    masked_concepts = []
    for i in range(batch_size):
        c_p = concept_positions[i]
        if len(c_p) == 0:
            continue
        
        #c_p = [c_p[1], c_p[2]]
        
        masked_c_p = []
        for p in c_p:
            #print(p)
            #p = [p[1], p[2]]
            #p = torch.Tensor(p).to(torch.long).cuda()
            
            if random.random() < masking_ratio:
                
                if masked_concept_freq != None:
                    masked_concept_freq[p[2]] += 1
                
                if p[1] > 128:
                    p[1] = 128
                
                if p[0] >= p[1]:
                    continue
                
                masked_concept_token += p[1] - p[0]
                
                #print(origin_input_ids)
                
                labels[i, p[0] : p[1]] = origin_input_ids[i, p[0] : p[1]] 
                concept_masked_position[i, p[0] : p[1]] = origin_input_ids[i, p[0] : p[1]] 
                if random.random() < 0.8:            
                    input_ids[i, p[0] : p[1]] = mask_tok
                elif random.random() < 0.5:    
                    replace = torch.randint(0, 31090, (p[1] - p[0],)).cuda()
                    #print(replace, p)
                    input_ids[i, p[0] : p[1]] = replace


    return input_ids, labels, masked_concept_token, concept_masked_position



def random_masking(origin_input_ids, input_ids, labels, concept_positions, masking_ratio = 0.4, masked_concept_freq = None):
    batch_size = input_ids.size(0)
    
    masked_concept_token = 0
    
    concept_masked_position = torch.zeros_like(labels).fill_(-100).cuda()
    for i in range(batch_size):
        
        masked_c_p = []
        for p in range(1, input_ids.shape[1] - 1):
            #print(p)
            #p = [p[1], p[2]]
            #p = torch.Tensor(p).to(torch.long).cuda()
            
            if random.random() < masking_ratio:
                
                #print(origin_input_ids)
                labels[i, p] = origin_input_ids[i, p] 
                if random.random() < 0.8:            
                    input_ids[i, p] = mask_tok
                elif random.random() < 0.5:    
                    replace = torch.randint(0, 31090, (p[1] - p[0],)).cuda()
                    #print(replace, p)
                    input_ids[i, p] = replace

    return input_ids, labels, masked_concept_token, concept_masked_position


def concept_masking_curriculum(origin_input_ids, input_ids, labels, concept_positions, mat_dict, masking_ratio = 0.4, masked_concept_freq = None):
    batch_size = input_ids.size(0)
    
    masked_concept_token = 0
    concept_masked_position = torch.zeros_like(labels).fill_(-100).cuda()
    
    #print(concept_positions)
    masked_concepts = []
    for i in range(batch_size):
        c_p = concept_positions[i]
        if len(c_p) == 0:
            continue
        
        #c_p = [c_p[1], c_p[2]]
        temp = []
        total_token_length = 0
        for p in c_p:
            if p[2] in mat_dict:
                temp.append(p)
                total_token_length += (p[1] - p[0])
        
        c_p = temp
        
        random.shuffle(c_p)

        for p in c_p:

            if random.random() < masking_ratio:
                
                if masked_concept_freq != None:
                    masked_concept_freq[p[2]] += 1
                
                if p[1] > 128:
                    p[1] = 128
                
                if p[0] >= p[1]:
                    continue
                
                masked_concept_token += p[1] - p[0]

                labels[i, p[0] : p[1]] = origin_input_ids[i, p[0] : p[1]] 
                concept_masked_position[i, p[0] : p[1]] = origin_input_ids[i, p[0] : p[1]] 
                if random.random() < 0.8:            
                    input_ids[i, p[0] : p[1]] = mask_tok
                elif random.random() < 0.5:    
                    replace = torch.randint(0, 31090, (p[1] - p[0],)).cuda()
                    input_ids[i, p[0] : p[1]] = replace


    return input_ids, concept_masked_position, masked_concept_token, concept_masked_position

        
model = BertForMaskedLM.from_pretrained(model_name, config = config)
model.resize_token_embeddings(len(tokenizer))

model.train()
data_collator = DataCollatorForWholeWordMask_custom(tokenizer = tokenizer, mlm_probability = args.raw_masking)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    'weight_decay': args.weight_decay},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr = args.lr, betas = (0.9, 0.98), eps = 1e-6)

criterion = torch.nn.CrossEntropyLoss(reduction = "none")
total_steps = 100000
warmup_ratio = 0.048

print(total_steps)

optimizer_scheduler = get_linear_schedule_with_warmup(
optimizer, num_warmup_steps=int(total_steps * warmup_ratio), 
num_training_steps=total_steps
)

criterion = torch.nn.CrossEntropyLoss(reduction = "none")

model.to(device)
scaler = torch.cuda.amp.GradScaler()

freq_list = []
wt_list = []

if "curriculum" in args.masking_strategy:
    print("curriculum_rel!")
    with open("/home/user10/MatSciBERT/data/all/concept_128_ngram_%s_similarity_%s_combine/curriculum_%d.pkl"%(args.concepts, args.similarity, args.curriculum_num), "rb") as f:
        curriculum_dict = pickle.load(f)
    
            
    for c in curriculum_dict:
        print(len(c))


masked_concept_freq = dict()
LM_LOSS = 0.
MASK_NUM = 0.
ACC = 0.
CONCEPT_ACC = 0.
num_loss = 0
iteration = 0
dynamic_p = 0
CONCEPT_MASK_NUM = 0.
for epoch in range(30):
    
    a = [i for i in range(10)]
    random.shuffle(a)
    
    for i in a:
        with open("/home/user10/MatSciBERT/data/all/concept_128_ngram_%s_similarity_%s_combine/split_real/train_data_%d.pkl"%(args.concepts, args.similarity, i+1), "rb") as f:  
            train_dataset = pickle.load(f)

        temp = dict()
        temp["input_ids"] = train_dataset["train_data"]
        temp["concept_positions"] = train_dataset["concepts"]
        train_dataset = temp

        train_dataset = MSC_Dataset_for_concept(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.step_batch_size, shuffle=True, collate_fn = data_collator, drop_last = True)
        

        for batch in tqdm(train_dataloader, ncols = 100, bar_format="{l_bar}{bar:25}{r_bar}"):
            
            iteration += 1
            
            #print(batch)
            step_whole_masked_input_ids = batch['input_ids'].to(device)
            step_labels = batch['labels'].to(device)
            step_concept_positions = batch['concept_positions']
            step_origin_input_ids = batch['origin_input_ids'].to(device)
            
            step_masked_input_ids = copy.deepcopy(batch['origin_input_ids']).to(device)
            

            if args.masking_strategy == "random":
                step_masked_input_ids, step_labels, masked_concept_token, concept_masked_position = concept_masking(step_origin_input_ids, step_masked_input_ids, step_labels, step_concept_positions, masking_ratio = args.masking_ratio, masked_concept_freq = masked_concept_freq)

            elif "curriculum" == args.masking_strategy:
            
                num = iteration // 10000
                num = num % (args.curriculum_num + 1)
                mat_dict = curriculum_dict[num - 1]

                curriculum_steps = (100000 // (args.curriculum_num + 1)) * (args.curriculum_num + 1)
                
                if iteration < curriculum_steps:

                    dynamic_p = args.masking_ratio + args.extra_masking
                    t = (num - 1) * (args.extra_masking / (args.curriculum_num - 1)) 
                    dynamic_p -= t
                        
                    if num == 0:
                        step_masked_input_ids = step_whole_masked_input_ids
                    elif num == args.curriculum_num:
                        step_masked_input_ids, step_labels, masked_concept_token, concept_masked_position = concept_masking(step_origin_input_ids, step_masked_input_ids, step_labels, step_concept_positions, masking_ratio = args.masking_ratio, masked_concept_freq = masked_concept_freq)
                    else:                    
                        step_masked_input_ids, step_labels, masked_concept_token, concept_masked_position = concept_masking_curriculum(step_origin_input_ids, step_masked_input_ids, step_labels, step_concept_positions, mat_dict, masking_ratio = dynamic_p, masked_concept_freq = masked_concept_freq)
                        
                else:
                    step_masked_input_ids, step_labels, masked_concept_token, concept_masked_position = concept_masking(step_origin_input_ids, step_masked_input_ids, step_labels, step_concept_positions, masking_ratio = args.masking_ratio, masked_concept_freq = masked_concept_freq)

            masked_token_num = len(step_masked_input_ids[step_masked_input_ids == mask_tok])

            for i in range(args.step_batch_size // args.batch_size):
            
                masked_input_ids = step_masked_input_ids[i * args.batch_size : (i+1) * args.batch_size]
                labels = step_labels[i * args.batch_size : (i+1) * args.batch_size]

                with torch.cuda.amp.autocast():
                    
                    outputs = model(input_ids = masked_input_ids, labels = labels)
                    logits = outputs.logits
                    loss = outputs.loss
                    loss = loss / (args.step_batch_size / args.batch_size)
                    loss = loss.mean()
                    
                logits = logits[labels != -100]
                labels = labels[labels != -100]
                
                predicted = torch.max(logits, dim = -1)[1]    
                
                    
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                LM_LOSS += loss.item()
                ACC += len(labels[labels == predicted]) / len(labels) / (args.step_batch_size / args.batch_size)
                
            scaler.step(optimizer)
            scaler.update()
            optimizer_scheduler.step()
            MASK_NUM += masked_token_num / args.step_batch_size
            num_loss += 1
                
            if iteration % 200 == 0:
                
                writer.add_scalar(f'Loss/LM_loss', LM_LOSS / (num_loss + 1e-8), iteration)
                writer.add_scalar(f'Loss/Mask_num', MASK_NUM / (num_loss + 1e-8), iteration)
                
                writer.add_scalar(f'Loss/LR', optimizer.param_groups[0]['lr'], iteration)
                writer.add_scalar(f'Loss/ACC', ACC / (num_loss + 1e-8), iteration)
                writer.add_scalar(f"Loss/Mask_ratio", dynamic_p, iteration)
                
                LM_LOSS = 0.
                MASK_NUM = 0.
                CONCEPT_MASK_NUM = 0.
                num_loss = 0.
                ACC = 0.
                CONCEPT_ACC = 0.

            if iteration % 10000 == 0:
                PATH = './model/MELT_128_%d_curri_%d_total_%f.pt' %(int(iteration), args.curriculum_num, round(args.masking_ratio,2))
                print("save the model")
                torch.save(model.state_dict(), PATH)

            if iteration > 100000:
                exit()
