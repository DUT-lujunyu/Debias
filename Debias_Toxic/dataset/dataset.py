import re
import os
import json
import time
import random
import torch
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer
# from Debias_Toxic.config.config import Config_base

def convert_onehot(config, label):
    onehot_label = [0 for i in range(config.num_classes)]
    onehot_label[int(label)] = 1
    return onehot_label

class Datasets(Dataset):
    '''
    The dataset based on Bert.
    '''
    def __init__(self, kwargs, data_name, if_adv=False, add_special_tokens=True, not_test=True):
        self.kwargs = kwargs
        self.device = kwargs.device
        self.if_adv = if_adv
        self.not_test = not_test
        self.data_name = data_name
        self.max_tok_len = kwargs.pad_size
        self.max_emb_tok_len = kwargs.emb_pad_size
        self.add_special_tokens = add_special_tokens  
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs.model_name)
        self.word_list_path = kwargs.word_list_path 
        with open(data_name, 'r') as f:
            self.data_file = json.load(f)
        self.preprocess_data()

        
    def preprocess_data(self):

        df = pd.read_csv(self.word_list_path)

        print('Preprocessing Data {} ...'.format(self.data_name))
        data_time_start=time.time()

        for row in tqdm(self.data_file):
            ori_text = row['tweet']
            row["one_hot_label"] = convert_onehot(self.kwargs, row["label"])
            
            # For Sentence Branch
            text = self.tokenizer(ori_text, add_special_tokens=self.add_special_tokens,
                                max_length=int(self.max_tok_len), padding='max_length', truncation=True)
            row['text_idx'] = text['input_ids']
            row['text_mask'] = text['attention_mask']
            
            # For Word Branch
            toxic_content = ' [SEP] '.join(row['toxic_content'])
            toxic_emb = self.tokenizer(toxic_content, add_special_tokens=self.add_special_tokens,
                                max_length=int(self.max_emb_tok_len), padding='max_length', truncation=True)    
            row['emb_idx'] = toxic_emb['input_ids']
            row['emb_mask'] = toxic_emb['attention_mask']

            # For Sentence-Word Branch
            row["text_emb_idx"] = text['input_ids'] + toxic_emb['input_ids']
            row["text_emb_mask"] = text['attention_mask'] + toxic_emb['attention_mask']

        data_time_end = time.time()
        print("... finished preprocessing cost {} ".format(data_time_end-data_time_start))

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx, corpus=None):
        row = self.data_file[idx]
        sample = {
                    # For Sentence Branch
                    'text_idx': row['text_idx'], 'text_mask': row['text_mask'],
                    # For Word Branch
                    'emb_idx': row['emb_idx'], 'emb_mask': row['emb_mask'],
                    # For Sentence-Word Branch
                    'text_emb_idx': row['text_emb_idx'], 'text_emb_mask': row['text_emb_mask'],
                }
        # For label
        sample['label'] = row['label']
        sample["one_hot_label"] = row["one_hot_label"]
        if not self.if_adv:
            sample['bias'] = row["bias"]
        return sample

    
class Dataloader(DataLoader):
    '''
    A batch sampler of a dataset. 
    '''
    def __init__(self, data, batch_size, shuffle=True, SEED=0):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle 
        self.SEED = SEED
        random.seed(self.SEED)

        self.indices = list(range(len(data))) 
        if shuffle:
            random.shuffle(self.indices) 
        self.batch_num = 0 

    def __len__(self):
        return int(len(self.data) / float(self.batch_size))

    def num_batches(self):
        return len(self.data) / float(self.batch_size)

    def __iter__(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.indices != []:
            idxs = self.indices[:self.batch_size]
            batch = [self.data.__getitem__(i) for i in idxs]
            self.indices = self.indices[self.batch_size:]
            return batch
        else:
            raise StopIteration

    def get(self):
        self.reset() 
        return self.__next__()

    def reset(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle: random.shuffle(self.indices)

            
def to_tensor(batch, if_adv=False):
    '''
    Convert a batch data into tensor
    '''
    args = {}

    # For label
    args["label"] = torch.tensor([b['label'] for b in batch])
    args["one_hot_label"] = torch.tensor([b['one_hot_label'] for b in batch])
    
    # For bias
    if not if_adv:
        args["bias"] = torch.tensor([b['bias'] for b in batch])

    # For Sentence Branch  
    args['text_idx'] = torch.tensor([b['text_idx'] for b in batch])
    args['text_mask'] = torch.tensor([b['text_mask'] for b in batch])

    # For Word Branch
    args['emb_idx'] = torch.tensor([b['emb_idx'] for b in batch])
    args['emb_mask'] = torch.tensor([b['emb_mask'] for b in batch])   

    # For Sentence-Word Branch
    args['text_emb_idx'] = torch.tensor([b['text_emb_idx'] for b in batch])
    args['text_emb_mask'] = torch.tensor([b['text_emb_mask'] for b in batch])      

    return args