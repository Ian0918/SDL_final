import os
import requests as re
from bs4 import BeautifulSoup
import asyncio, aiohttp

DATA_PATH_PREFIX = '../raw'
# if need to start the server within the process, set START_SERVER=True;
# if you init for the first time, set INIT = True
# START_SERVER = True
# INIT = False

async def getSub(mov_title, session):
    async with session.get(f'http://localhost:3000/api/eng/search?q={"+".join(mov_title.lower().split(" "))}&type=movies') as res:
        if res.status == 200:
            json = await res.json()
            print(mov_title, len(json), json[0]['url'])
            async with session.get(json[0]['url']) as movie_res:
                if movie_res.status == 200:
                    res_txt = await movie_res.text()
                    soup = BeautifulSoup(res_txt, features='html.parser')
                    prefix = 'https://www.opensubtitles.org'
                    postfix = soup.findAll(class_='bnone')
                    print(f'movie: {mov_title}, postfix: {postfix}')
                    postfix = postfix[0]['href']
                    async with session.get(f'{prefix}/en/subtitleserve/sub/{postfix.split("/")[-2]}') as dl_page:
                        if dl_page.status == 200:
                            if not os.path.isdir(f'{DATA_PATH_PREFIX}/../sub'):
                                os.mkdir(f'{DATA_PATH_PREFIX}/../sub')
                            with open(f'{DATA_PATH_PREFIX}/../sub/{mov_title}.zip', 'wb') as f:
                                dl_file = await dl_page.content.read()
                                f.write(dl_file)
                                return True
                        else:
                            return False
                else:
                    return False
        else:
            return False

async def main(MVs):
    async with aiohttp.ClientSession() as session: 
        tasks = []
        for i, mv in enumerate(MVs):
            tasks.append(getSub(mv, session=session))
            # if i % 10 == 0:
            #     await asyncio.sleep(1)
        await asyncio.gather(*tasks)

if __name__ == '__main__':

    # if START_SERVER:
    #     import subprocess
    #     if INIT:
    #         subprocess.run("cd ./Opensubtitles-Unofficial-API && npm install && npm start".split())
    #     else:
    #         subprocess.run("cd ./Opensubtitles-Unofficial-API && npm start".split())

    import pandas as pd
    mv_df = pd.read_csv(f'{DATA_PATH_PREFIX}/movies_metadata.csv')
    enMv_titles = mv_df[mv_df['original_language'] == 'en']['original_title']
    enMv_titles = enMv_titles[:3]
    
    asyncio.run(main(enMv_titles))