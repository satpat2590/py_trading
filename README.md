# Python Stock Aggregator & Options Technical Analysis Program 

## Set up virtual environment

Currently, the directory runs via a 'venv' setup using Python's internal venv module. <b>Soon<b> this will be uploaded as a Docker image for easier
access and worry-free dependency configuration. 

### venv setup (for Unix-based systems [MacOS, Linux])
1. run the following command to create a python virtual environment

```bash
python3 -m venv [path/to/file] 
```

2. activate the virtual environment by running the 'activate' command within the ~/bin folder

```bash
source ./bin/activate
```

3. once done with the environment, you can shut it down running the following

```bash
deactivate
```

## Run the main script 

Run the financials.py script to start the real-time data stream. 
```bash
python trading/financials.py
```


## Additionals remaining... 

- store EOD stock-data and options-chain data to a file on your system. 

- modify OptionsTrading class to pull from .csv file and initialize OptionsTrading.stock_data & OptionsTrading.options_data 
DataFrames. 

- run sigma calculations using historical stock data. 
    - calculate mean and standard deviation of stocks based on tickers first

- find 1% yield of each stock, embed them into a final DataFrame which takes in calculations and options information 

- use financial graph API to plot and show tabular data of the DataFrames in a terminal-based UI 
