import os, json
import pandas as pd

DATA_PATH_PREFIX = '..'
mv_df = pd.read_csv(f'{DATA_PATH_PREFIX}/raw/movies_metadata.csv')
enMv_titles = mv_df[mv_df['original_language'] == 'en']['original_title']
MVs = {tit: {'status': "unready", 'file': ""} for tit in enMv_titles}

readys = {m.split('.')[0]: {'status': 'ready', 'file': m} for m in os.listdir(f'{DATA_PATH_PREFIX}/sub') if m.split('.')[-1] == 'zip'}
for mv in readys:
    MVs[mv] = readys[mv]
with open(f'{DATA_PATH_PREFIX}/sub/scrap_status.json', 'w') as f:
    json.dump(MVs, f)