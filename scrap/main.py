import os, json
import requests as re
from bs4 import BeautifulSoup
import asyncio, aiohttp
from tqdm.asyncio import tqdm_asyncio
import logging
import random

DATA_PATH_PREFIX = '..'
# if need to start the server within the process, set START_SERVER=True;
# if you init for the first time, set INIT = True
# START_SERVER = True
# INIT = False

proxies = [
    'https://195.181.134.36:3128',
    'https://81.134.57.82:3128'
]

async def getSub(mov_title, session, log):
    def delay_time(res):
        if (res.status == 429):
            logging.info(f'Retry-After: {res.headers["Retry-After"] if "Retry-After" in res.headers else "IDK"}\n')

    fail_status = {'status': 'unready', 'file': ''}
    async with session.get(f'http://localhost:3000/api/eng/search?q={"+".join(mov_title.lower().split(" "))}&type=movies') as res:
        if res.status == 200:
            res_json = await res.json()
        else:
            fail_status['status'] = f'{res.status}'
            delay_time(res)
            return fail_status

    logging.info(f'{mov_title} {len(res_json)}')
    cur_status = ""
    for i in range(len(res_json)):
        if not res_json[i]:
            continue
        logger.info(f'{res_json[i]["url"]}')
        async with session.get(res_json[i]['url']) as movie_res:
            if movie_res.status == 200:
                res_txt = await movie_res.text()
                soup = BeautifulSoup(res_txt, features='html.parser')
            else:
                cur_status = movie_res.status
                logging.warning(f'{mov_title} {res_json[i]["url"]} response status {movie_res.status}')
                delay_time(movie_res)
                continue
            if len(soup.findAll(id='search_results')) == 0:
                prefix = 'https://www.opensubtitles.org'
                # postfixs = [res_json[i]['url'].split('/')[-2]]
                postfixs = [res_json[i]['url']]
            else:
                prefix = 'https://www.opensubtitles.org'
                postfixs = [p['href'] for p in soup.findAll(class_='bnone')]

        for i in range(len(postfixs)):
            postfix = postfixs[i].split('/')[-2]
            logging.info(f'movie: {mov_title}, postfix: {postfix}')

            async with session.get(f'{prefix}/en/subtitleserve/sub/{postfix}', proxy=random.choice(proxies), verify=False) as dl_page:
                #await dl_page.status
                if dl_page.status == 200:
                    if not os.path.isdir(f'{DATA_PATH_PREFIX}/sub'):
                        os.mkdir(f'{DATA_PATH_PREFIX}/sub')
                    with open(f'{DATA_PATH_PREFIX}/sub/{mov_title}.zip', 'wb') as f:
                        dl_file = await dl_page.content.read()
                        f.write(dl_file)
                    logging.info(f'wrote movie {mov_title}')           
                    return {'status': 'ready', 'file': f'{DATA_PATH_PREFIX}/sub/{mov_title}.zip'}
                else:
                    cur_status = dl_page.status
                    logging.warning(f'{mov_title} {f"{prefix}/en/subtitleserve/sub/{postfix}"} response status {dl_page.status}')
                    delay_time(dl_page)
                    continue
    fail_status['status'] = f'{cur_status}'
    return fail_status
        
async def semSub(semaphore, mv, session, log=None):
    async with semaphore:
        # try:
        #     res = await getSub(mv, session, log)
        # except Exception as e:
        #     res = {'status': f'{f"{e}: {e.message}" if hasattr(e, "message") else f"{e}"}', 'file': ''}
        res = await getSub(mv, session, log)
        logging.info(f'{mv}: {res}')
        return {mv: res}

async def main(MVs: dict):
    sem = asyncio.Semaphore(3)
    agents = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    ]
    async with aiohttp.ClientSession(headers={'User-Agent': agents[0]}) as session: 
        tasks = []
        for i, mv in enumerate(MVs):
            if MVs[mv]['status'] != 'ready':
                tasks.append(semSub(sem, mv, session=session))
        return await tqdm_asyncio.gather(*tasks)

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

    #logging.basicConfig(level=logging.info, filemode='a', filename='progress.log')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s %(levelname)-8s] %(message)s', datefmt='%Y%m%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    fh = logging.FileHandler('progress.log')
    fh.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    #logger.addHandler(ch)
    logger.addHandler(fh)

    if not os.path.isfile(f'{DATA_PATH_PREFIX}/sub/scrap_status.json'):
        import pandas as pd
        mv_df = pd.read_csv(f'{DATA_PATH_PREFIX}/raw/movies_metadata.csv')
        enMv_titles = mv_df[mv_df['original_language'] == 'en']['original_title']
        MVs = {tit: {'status': "", 'file': ""} for tit in enMv_titles}
    else:
        with open(f'{DATA_PATH_PREFIX}/sub/scrap_status.json') as f:
            MVs = json.load(f)
    res = asyncio.run(main(MVs))
    for r in res:
        MVs.update(r)
    with open(f'{DATA_PATH_PREFIX}/sub/scrap_status.json', 'w') as f:
        json.dump(MVs, f)

