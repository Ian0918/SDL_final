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
            all_tok = self.tokenizer(sentence, return_tensors='pt')
            head_tok = {
                'input_ids': all_tok['input_ids'][:, :129].clone().detach(),
                'token_type_ids': all_tok['token_type_ids'][:, :129].clone().detach(),
                'attention_mask': all_tok['attention_mask'][:, :129].clone().detach()
            }
            tail_tok = {
                'input_ids': all_tok['input_ids'][:, len(all_tok['input_ids']) - (512 - 129):].clone().detach(),
                'token_type_ids': all_tok['token_type_ids'][:, len(all_tok['token_type_ids']) - (512 - 129):].clone().detach(),
                'attention_mask': all_tok['attention_mask'][:, len(all_tok['attention_mask']) - (512 - 129):].clone().detach()
            }
            tok = {
                'input_ids': torch.cat((head_tok['input_ids'], tail_tok['input_ids']), dim=1).to(self.device),
                'token_type_ids': torch.cat((head_tok['token_type_ids'], tail_tok['token_type_ids']), dim=1).to(self.device),
                'attention_mask': torch.cat((head_tok['attention_mask'], tail_tok['attention_mask']), dim=1).to(self.device)
            }
            return self.model(**tok)[0][:, 0, :].reshape(-1).to('cpu')
        elif self.method[0] == "chunk":
            pass
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