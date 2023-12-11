import os, json
import gc
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from model import LongBERT

def prepare_model(modelname = 'bert-base-uncased'):
    model = AutoModel.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    
    return model, tokenizer

def prepare_dataset(data_path):
    assert os.path.isfile(data_path)
    return pd.read_parquet(data_path)

def get_longsubs(data=None):
    '''
    data: pd.DataFrame, which is gotten from prepare_dataset()

    return: pd.Series, each row is a full text of `Context` column in `data`
    '''
    def sub_concat(subs):
        subs = json.loads(subs)
        long_sub = ""
        for sub in subs:
            long_sub += sub['Text']
        return long_sub
    assert not data is None
    long_subs = data['Context'].apply(sub_concat)
    long_subs.reset_index(drop=True, inplace=True)
    return long_subs

def inference_bert(data, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    '''
    data: array-like, each row is a sentence to infer by BERT
    model: transformers.Model, the model instance used for inference
    tokenizer: transformers.Tokenizer, the tokenizer used for inference
    device: str, 'cuda' or 'cpu', default 'cuda' if torch.cuda.is_available() else 'cpu'

    return: list, BERT CLS token of every row in `data`
    '''
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

def k_means(x, **kwargs):
    '''
    x: array-like object, that contains data for clustering
    **kwargs: any parameter for sklearn.cluster.KMeans object
    '''
    from sklearn.cluster import KMeans

    knn = KMeans(**kwargs)
    if isinstance(x, np.ndarray):
        knn.fit(x)
    else:
        try:
            knn.fit(np.array(x))
        except:
            raise TypeError("\'x\' is not an array-like object")
    return knn.labels_
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", "-m", type=str, default="bert-base-cased")
    parser.add_argument("--method", type=str, default="head")
    parser.add_argument("--data", "-d", type=str, default="data/exp_data.parquet")
    parser.add_argument("--save", "-s", type=bool, default=False)

    args = parser.parse_args()

    assert args.method in ['head', 'head_tail', 'chunk']

    model, tokenizer = prepare_model(modelname=args.modelname)
    data = prepare_dataset(data_path=args.data)
    long_subs = get_longsubs(data)
    LB = LongBERT({"model": model, "tokenizer": tokenizer, "method": "head_tail"})
    
    movie_cls_token = LB.inference(long_subs)

    print(movie_cls_token)

    if args.save:
        pd.DataFrame(movie_cls_token).to_parquet(f"{args.modelname}_{args.method}.parquet")

    # add your code to manipulate `movie_cls_token` here
    # print(k_means(movie_cls_token, n_clusters=8))