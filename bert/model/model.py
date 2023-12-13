import torch
from torch import nn
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc


class LongBERT(nn.Module):

    def __init__(self, conf = None):
        super().__init__()

        if conf and 'method' in conf:
            self.method = conf['method'].split()
        else:
            self.method = ['head']

        if conf and 'device' in conf:
            self.device = conf['device']
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if conf and 'model' in conf:
            self.model = conf['model'].to(self.device)
        else:
            self.model = AutoModel("bert-base-uncased").to(self.device)

        if conf and 'tokenizer' in conf:
            self.tokenizer = conf['tokenizer']
        else:
            self.tokenizer = AutoTokenizer("bert-base-uncased")


    def _cls(self, sentence):
        
        if self.method[0] == "head":
            return self.model(**self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(self.device))[0][:, 0, :].reshape(-1).to('cpu')
        elif self.method[0] == "head_tail":
            if len(self.method) >= 2:
                head_num = self.method[1]
            else:
                head_num = int(0.2 * self.tokenizer.model_max_length) + 1
            all_tok = self.tokenizer(sentence, return_tensors='pt')
            head_tok = {k: all_tok[k][:, :head_num].clone().detach() for k in all_tok.keys()}
            tail_tok = {k: all_tok[k][:, len(all_tok[k])- (self.tokenizer.model_max_length - head_num):].clone().detach() for k in all_tok.keys()}
            tok = {k: torch.cat((head_tok[k], tail_tok[k]), dim=1).to(self.device) for k in all_tok.keys()}
            return self.model(**tok)[0][:, 0, :].reshape(-1).to('cpu')
        elif self.method[0] == "chunk":
            raise NotImplementedError()
        else:
            return None

    def inference(self, x):

        cls_sub = []
        with torch.no_grad():
            for sub in tqdm(x):
                cls_tok = self._cls(sub)
                cls_sub.append(cls_tok)
            
                #del sub_enc, output
        torch.cuda.empty_cache()
        gc.collect()
        return np.array(cls_sub)


# if __name__ == "__main__":
    
#     import pandas as pd

#     model, tokenizer = main.prepare_model(modelname=args.modelname)
#     data = main.prepare_dataset()
#     long_subs = main.get_longsubs(data)

#     LB = LongBERT({"model": model, "tokenizer": tokenizer, "method": "head_tail"})
    
#     movie_cls_token = LB.inference(long_subs)
#     print(movie_cls_token)