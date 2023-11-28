# SDL_final

## Usage
1. Clone from github repo
For those who haven't clone the repo before:
```{shell}
git clone --recursive https://github.com/Ian0918/SDL_final.git
```

For those that already clone the repo but need to pull updates:
If you have some local commits:
```{shell}
git pull --rebase
```
, or not:
```{shell}
git pull
```
If yo need to use submodules (in this repo, `Opensubtitles-Unofficial-API`):
```{shell}
git pull --rebase --recurse-submodules && git submodule update --recursive --init
```

2. (Optional) Unzip raw data
For those who have to perform with the raw data (you have to download the raw data into `raw` in advance),
```{shell}
unzip raw/the-movies-dataset.zip
```

3. Environment setup
```{shell}
# SDL_final
python -m pip install -r requirements.txt
```
(Optional but recommended)
`git-lfs`: [official site](https://git-lfs.com/)

## TODOs:
- [x] scrap subtitle function
- [x] read dataset & demo 3 subs
- [x] async scrap (TODO: edge cases & 429 error)
- [ ] scrap all related movies' subtitle
