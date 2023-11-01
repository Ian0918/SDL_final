# SDL_final

## Usage
1. Clone from github repo
For those who haven't clone the repo before:
```{shell}
git clone --recursive https://github.com/Ian0918/SDL_final.git
```

For those that already clone the repo but need to pull updates:
```{shell}
git pull --rebase --recurse-submodules && git submodule update --recursive --init
```

2. (Optional) Unzip raw data
For those who have to perform with the raw data,
```{shell}
unzip raw/the-movies-dataset.zip
```

## TODOs:
- [x] scrap subtitle function
- [x] read dataset & demo 3 subs
- [ ] async scrap (TODO: edge cases & 429 error)
- [ ] scrap all related movies' subtitle
