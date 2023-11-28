import os, json
import gc
import tqdm

import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer

def prepare_model():
    model = BertModel('bert-base-uncased')
    tokenizer = BertTokenizer('bert-base-uncased')
    
    return model, tokenizer

def prepare_dataset():
    return pd.read_parquet('data/exp_data.parquet')

def get_longsubs(data=None):
    def sub_concat(subs):
        subs = json.loads(subs)
        long_sub = ""
        for sub in subs:
            long_sub += sub['Text']
        return long_sub
    assert data
    long_subs = data['Context'].apply(sub_concat)
    long_subs.reset_index(drop=True, inplace=True)
    return long_subs

def inference_bert(data, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    
    cls_sub = []
    with torch.no_grad():
        for sub in tqdm(data):
            sub_enc = tokenizer(sub, return_tensors='pt', padding=True, truncation=True).to(device)
            output = model(**sub_enc)
            cls_tok = output[0][:, 0, :].reshape(-1).to('cpu')
            cls_sub.append(cls_tok)
            
            #sub_enc.to('cpu')
            del sub_enc, output
        torch.cuda.empty_cache()
        gc.collect()
    return cls_sub

if __name__ == "__main__":
    model, tokenizer = prepare_model()
    data = prepare_dataset()
    long_subs = get_longsubs(data)
    movie_cls_token = inference_bert(long_subs, model, tokenizer)