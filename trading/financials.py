"""
Module for connecting to a Finance Websocket and retrieving data from it
"""
# import pandas as pd 
# import numpy as np 
# import yfinance as yf 
import yliveticker 
import pandas as pd 
import numpy as np 
import yfinance as yf 
from collections import defaultdict, deque
from functools import partial 
import datetime


def options_chain(symbol):
    tk = yf.Ticker(symbol) 
    # Expiration dates
    exps = tk.options
    options = pd.DataFrame()
    for e in exps: 
        opt = tk.option_chain(e)
        opt = pd.DataFrame()._append(opt.calls)._append(opt.puts)
        opt['expirationDate'] = e
        options = options._append(opt, ignore_index=True)
    
    # Bizarre error in yfinance gives wrong expiration date
    # Add 1 day to get the correct expiration date
    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days=1) 
    options['dte'] = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365

    # Boolean column if the option is a CALL 
    options['CALL'] = options['contractSymbol'].str[4:].apply(
        lambda x: "C" in x)

    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / 2 # Calculate midpoint of the bid-ask

    # Drop unnecessary and meaningless columns
    options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])
    # Return options chain 
    return options

def on_new_msg(ws, msg):
    print(msg)

apple_chain = options_chain('AAPL')
print(apple_chain[apple_chain['CALL'] == False])

yliveticker.YLiveTicker(on_ticker=on_new_msg, ticker_names=[
    "AAPL"
])


