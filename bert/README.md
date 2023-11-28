# BERT CLS token inference

## Usage
```{shell}
python main.py
```
## Code Structure
`main.py`:
```{python}

def prepare_model():
    # get model and tokenizer
    ...

def prepare_dataset():
    # load and return dataset
    ...

def get_longsubs(data=None):
    # returns an array of long subs from dataset `data`
    ...

def inference_bert(data, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # returns an array CLS of every data in `data`, using `model` and `tokenizer`
    ...
```