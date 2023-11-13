import os, json

DATA_PATH = "../sub"

if not os.path.isdir(DATA_PATH):
    raise Exception(f'DATA_PATH {DATA_PATH} is not a dir.')

if not os.path.isfile(f'{DATA_PATH}/scrap_status.json'):
    raise Exception(f'Cannot find file {f"{DATA_PATH}/scrap_status.json"}')

with open(f'{DATA_PATH}/scrap_status.json') as f:
    mv_stat = json.load(f)

all_ready = True
cnt = 0
for mv in mv_stat:
    if mv_stat[mv]['status'] != "ready":
        all_ready = False
    else:
        cnt += 1
print(cnt, all_ready)