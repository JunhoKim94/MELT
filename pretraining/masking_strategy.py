from torch.utils.data import Dataset
from transformers import DataCollatorForWholeWordMask, BertTokenizer, BertTokenizerFast
import pickle
import random
import copy
import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
import numpy as np
import warnings

def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()

def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = min(max(x.size(0) for x in examples), 512)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

class DataCollatorForWholeWordMask_custom(DataCollatorForWholeWordMask):

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:

        if isinstance(examples[0], Mapping):
            concept_position = [e['concept_positions'] for e in examples]
            input_ids = [e["input_ids"] for e in examples]
        else:
        
            concept_position = [e['concept_positions'] for e in examples]
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        real_batch_input = copy.deepcopy(batch_input)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            #print(ref_tokens)
            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            mask_labels.append(self._whole_word_mask(ref_tokens))
            
        #print(mask_labels)
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)
        return {"origin_input_ids": real_batch_input, "input_ids": inputs, "labels": labels, "concept_positions" : concept_position}


    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        import torch

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class DataCollatorForWholeWordMask_custom_maskratio(DataCollatorForWholeWordMask):

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:

        if isinstance(examples[0], Mapping):
            concept_position = [e['concept_positions'] for e in examples]
            input_ids = [e["input_ids"] for e in examples]
        else:
        
            concept_position = [e['concept_positions'] for e in examples]
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        real_batch_input = copy.deepcopy(batch_input)

        mask_labels = []
        
        real_masked_tokens = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            #print(ref_tokens)
            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
                        
            whole_masked, masked_tokens = self._whole_word_mask(ref_tokens)
            mask_labels.append(whole_masked)
            real_masked_tokens.append(masked_tokens)
            
        #print(mask_labels)
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)
        return {"origin_input_ids": real_batch_input, "input_ids": inputs, "labels": labels, "concept_positions" : concept_position, "masked_tokens": real_masked_tokens}

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        cand_indexes = []
        
        word_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
                word_indexes[-1].append(token)
                
            else:
                cand_indexes.append([i])
                word_indexes.append([token])

        random_indices = [i for i in range(len(cand_indexes))]
        random.shuffle(random_indices)

        temp = []
        for s in word_indexes:
            #temp.append(self.tokenizer.decode(s, skip_special_tokens=True))
            if len(s) > 1:
                string = s[0]
                for w in s[1:]:
                    string += w[2:]
                #string.replace("##", "")
                temp.append(string)
            else:
                temp.append(s[0])
                
        word_indexes = temp

        t1 = []
        t2 = []
        for i in random_indices:
            t1.append(cand_indexes[i])
            t2.append(word_indexes[i])

        cand_indexes = t1
        word_indexes = t2
        #print(word_indexes)
        #random.shuffle(cand_indexes)
        #print(cand_indexes)
        
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        
        masked_tokens = []
        covered_indexes = set()
        for index_set, token_set in zip(cand_indexes,word_indexes):
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)
            masked_tokens.append(token_set)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        #print("cover", covered_indexes)
        return mask_labels, masked_tokens

    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        import torch

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

'''
class create_dataset(Dataset):
    def __init__(self, mode, tokenizer, path):
        print("create_dataset..." + mode)
        with open(path + mode, "rb") as f:
            self.datas = pickle.load(f)
        self.tokenizer = tokenizer
    def __len__(self):
        # 데이터 전체의 사이즈 반환
        return len(self.datas)
        
    def random_masking(self, cpt_masked_sentence, adjusting_mask_prob, mask_count):
        masked_sentence = []
        label_mask_ = []
        lm_position = []
        for id_position, id in enumerate(cpt_masked_sentence):
            if id not in [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                          self.tokenizer.mask_token_id]:
                if random.random() <= adjusting_mask_prob:
                    lm_position.append(id_position+1)
                    mask_count += 1
                    label_mask_.append(False)  # masking 할거면 false
                    if random.random() >= 0.2:
                        masked_sentence.append(self.tokenizer.mask_token_id)
                    elif random.random() <= 0.5:
                        masked_sentence.append(random.randint(1, 30521))
                    else:
                        masked_sentence.append(id)
                else:
                    label_mask_.append(True)
                    masked_sentence.append(id)
            else:
                if id == self.tokenizer.mask_token_id:
                    label_mask_.append(False)  # 이미 mask token이면 false
                else:
                    label_mask_.append(True)
                masked_sentence.append(id)

        return masked_sentence, torch.BoolTensor(label_mask_), lm_position, mask_count

    def __getitem__(self, idx):
        #datapoint = self.datas[idx]
        # 학습할 corpus에 있는 concept 찾고
        datapoint = self.datas[idx]
        mask_count = 0
        masked_sentence, label_mask_, lm_position, mask_count = self.random_masking(datapoint['encoded_txt'], 0.15, mask_count)
        datapoint['masking_txt'] = masked_sentence
        datapoint['label_mask'] = label_mask_.tolist()
        #datapoint['lm_position'] = lm_position
        return datapoint

def padded_sequence(samples):
        
    masked_LM = []
    LM_label = []
    label_mask_ = []
    label_position = []
    LM_max = 0
    for sample in samples:
        masked_LM.append(sample['masking_txt'])
        LM_label.append(sample['encoded_txt'])
        label_mask_.append(sample['label_mask'])
        label_position.append(sample['lm_position'])
        if len(sample['masking_txt']) > LM_max:
            LM_max = len(sample['masking_txt'])

    # print("q_max, a_max:", q_max, a_max)
    masked_lm_batch = []
    lm_label_batch = []
    label_mask_batch = []
    if LM_max > 128:
        LM_max = 128
    for i, LM_example in enumerate(masked_LM):
        if len(LM_example) <= LM_max:
            masked_lm_batch.append([tokenizer.cls_token_id]+LM_example+[tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (LM_max-len(LM_example)))
            lm_label_batch.append([tokenizer.cls_token_id]+LM_label[i]+[tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (LM_max-len(LM_label[i])))
            label_mask_batch.append([True]+label_mask_[i]+[True]+[True]*(LM_max-len(label_mask_[i])))
        else:
            masked_lm_batch.append(
                [tokenizer.cls_token_id] + LM_example[:LM_max] + [tokenizer.sep_token_id])
            lm_label_batch.append([tokenizer.cls_token_id]+ LM_label[i][:LM_max] + [tokenizer.sep_token_id])
            label_mask_batch.append([True] + label_mask_[i][:LM_max] + [True])

    return masked_lm_batch, lm_label_batch, label_mask_batch, label_position
'''