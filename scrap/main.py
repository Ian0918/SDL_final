import os, json
import requests as re
from bs4 import BeautifulSoup
import asyncio, aiohttp

DATA_PATH_PREFIX = '..'
# if need to start the server within the process, set START_SERVER=True;
# if you init for the first time, set INIT = True
# START_SERVER = True
# INIT = False

async def getSub(mov_title, session, log):
    fail_status = {'status': 'unready', 'file': ''}
    async with session.get(f'http://localhost:3000/api/eng/search?q={"+".join(mov_title.lower().split(" "))}&type=movies') as res:
        if res.status == 200:
            res_json = await res.json()
        else:
            return fail_status

    print(mov_title, len(res_json),res_json[0]['url'])
    async with session.get(res_json[0]['url']) as movie_res:
        if movie_res.status == 200:
            res_txt = await movie_res.text()
            soup = BeautifulSoup(res_txt, features='html.parser')
        else:
            return fail_status
        if len(soup.find(id='search_results')) == 0:
            prefix = 'https://www.opensubtitles.org'
            postfix = res_json[0]['url'].split('/')[-2]
        else:
            prefix = 'https://www.opensubtitles.org'
            postfix = soup.findAll(class_='bnone')
            postfix = postfix[0]['href'].split('/')[-2]
        print(f'movie: {mov_title}, postfix: {postfix}')

    async with session.get(f'{prefix}/en/subtitleserve/sub/{postfix}') as dl_page:
        if dl_page.status == 200:
            if not os.path.isdir(f'{DATA_PATH_PREFIX}/sub'):
                os.mkdir(f'{DATA_PATH_PREFIX}/sub')
            with open(f'{DATA_PATH_PREFIX}/sub/{mov_title}.zip', 'wb') as f:
                dl_file = await dl_page.content.read()
                f.write(dl_file)
            return {'status': 'ready', 'file': f'{DATA_PATH_PREFIX}/sub/{mov_title}.zip'}
        else:
            return fail_status
        
async def semSub(semaphore, mv, session, log=None):
    async with semaphore:
        try:
            res = await getSub(mv, session, log)
        except:
            res = {'status': 'unready', 'file': ''}
        return {mv: res}

async def main(MVs):
    sem = asyncio.Semaphore(3)
    async with aiohttp.ClientSession() as session: 
        tasks = []
        for i, mv in enumerate(MVs):
            if MVs[mv]['status'] != 'ready':
                tasks.append(semSub(sem, mv, session=session))
            # if i % 10 == 0:
            #     await asyncio.sleep(1)
        return await asyncio.gather(*tasks)

if __name__ == '__main__':

    # if START_SERVER:
    #     import subprocess
    #     if INIT:
    #         subprocess.run("cd ./Opensubtitles-Unofficial-API && npm install && npm start".split())
    #     else:
    #         subprocess.run("cd ./Opensubtitles-Unofficial-API && npm start".split())

    # if os.path.isdir(f'{DATA_PATH_PREFIX}/sub') and not os.path.isfile(f'{DATA_PATH_PREFIX}/sub/scrap_status.json'):
    #     MVs = [m.split('.')[0] for m in os.listdir(f'{DATA_PATH_PREFIX}/sub') if m.split('.')[-1] == 'zip']
    #     with open(f'{DATA_PATH_PREFIX}/sub/scrap_status.json', 'w') as f:
    #         json.dump(MVs, f)

    if not os.path.isfile(f'{DATA_PATH_PREFIX}/sub/scrap_status.json'):
        import pandas as pd
        mv_df = pd.read_csv(f'{DATA_PATH_PREFIX}/raw/movies_metadata.csv')
        enMv_titles = mv_df[mv_df['original_language'] == 'en']['original_title']
        MVs = {tit: {'status': "", 'file': ""} for tit in enMv_titles}
    else:
        with open(f'{DATA_PATH_PREFIX}/sub/scrap_status.json') as f:
            MVs = json.load(f)
    enMv_titles = MVs.keys()
    try:
        res = asyncio.run(main(enMv_titles))
    finally:
        for r in res:
            MVs[r] = res[r]
        with open(f'{DATA_PATH_PREFIX}/sub/scrap_status.json', 'w') as f:
            json.dump(MVs, f)