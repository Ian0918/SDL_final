# Scrap subtitle

## Usage
### Init setup

1. run the server with shell
```{shell}
cd Opensubtitles-Unoffcial-API && npm install && npm start
```


### After setting up for the first time
1. run the server with shell
```{shell}
cd Opensubtitles-Unoffcial-API && npm start
```

### Scraping
1. `main.py`: automatically start scraping and store the result into `../sub`, and log into `progress.log`
```{shell}
python main.py
```


2. `fix.py`: update `../sub/scrap_status.json` if terminated `main` unnaturally (unexpected error, keyboard interupt, ...)
```{shell}
python fix.py
```

3. `check.py`: show how many movies have already been scrapped and the files are ready 
```{shell}
python check.py
```
