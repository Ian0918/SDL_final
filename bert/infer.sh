python main.py -m "bert-base-uncased" --method "head" -d "data/exp_data2.parquet" -s True
python main.py -m "bert-base-uncased" --method "head_tail" -d "data/exp_data2.parquet" -s True
python main.py -m "bert-large-uncased" --method "head" -d "data/exp_data2.parquet" -s True
python main.py -m "bert-large-uncased" --method "head_tail" -d "data/exp_data2.parquet" -s True
python main.py -m "roberta-base" --method "head" -d "data/exp_data2.parquet" -s True
python main.py -m "roberta-base" --method "head_tail" -d "data/exp_data2.parquet" -s True
python main.py -m "roberta-large" --method "head" -d "data/exp_data2.parquet" -s True
python main.py -m "roberta-large" --method "head_tail" -d "data/exp_data2.parquet" -s True
