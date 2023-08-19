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
from scipy.stats import norm
import math

# Options Trading class incorporating real time stock data and a collection of options chains 
class OptionsTrading(): 
    def __init__(self):
        # Initialize stock_data DataFrame 
        stock_data_columns = ['timestamp', 'price', 'changePercent', 'dayVolume']
        self.stock_data = pd.DataFrame(columns=stock_data_columns)

        # Initialize the stock tickers you need to analyze
        self.ticker_data = ['AAPL']

        # Initialize all of the options chain data needed
        self.options_data = {x: options_chain(x) for x in self.ticker_data}




def calculate_delta_theta(row, risk_free_rate=0.01, dividend_yield=0.0):
    """
    Function to calculate the Delta and Theta of a particular option. 
    This function is calculated on each option through the pd.apply method. 

    The output is then spit out as a new column within the options DataFrame. 

    Delta: Rate of change of the option price relative to the underlying stock price.

    Theta: Time decay of the option contract, with the price decreasing relative to the time to expiration. 

    :param row: A single option and its corresponding feature data  
    :param risk_free_rate: A constant value that is decided strategically for the delta function. You can choose to alter the constant within 
                            the function itself. 
    :param dividend_yield: TBD
    :return: A pandas Series containing delta and theta which will get merged into the main options chain DataFrame 
    """
    S = row['price']  # Underlying asset price
    K = row['strike'] # Strike price
    t = row['dte']    # Time to expiration in years
    sigma = row['impliedVolatility'] # Implied volatility
    q = dividend_yield # Dividend yield

    # Calculate d1 and d2
    d1 = (math.log(S / K) + (risk_free_rate - q + (sigma ** 2) / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    # Calculate Delta
    if row['CALL']:
        delta = math.exp(-q * t) * norm.cdf(d1)
    else: # Put option
        delta = -math.exp(-q * t) * norm.cdf(-d1)

    # Calculate Theta
    theta_part1 = -(S * sigma * math.exp(-q * t) * norm.pdf(d1)) / (2 * math.sqrt(t))
    theta_part2 = risk_free_rate * K * math.exp(-risk_free_rate * t) * norm.cdf(d2 if row['CALL'] else -d2)
    theta_part3 = q * S * math.exp(-q * t) * norm.cdf(d1 if row['CALL'] else -d1)
    
    theta = theta_part1 - theta_part2 + theta_part3 if row['CALL'] else theta_part1 + theta_part2 - theta_part3

    return pd.Series({'delta': delta, 'theta': theta})


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

    global stock_data 
    timestamp = msg['timestamp']
    price = msg['price']
    change_percent = msg['changePercent']
    day_volume = msg['dayVolume']

    stock_data = stock_data.append({
        'timestamp': timestamp,
        'price': price,
        'changePercent': change_percent,
        'dayVolume': day_volume,
    }, ignore_index=True)


apple_chain = options_chain('AAPL')

# Printing out only the PUT options
print(apple_chain[apple_chain['CALL'] == False])

yliveticker.YLiveTicker(on_ticker=on_new_msg, ticker_names=[
    "AAPL"
])

# Assuming 'stock_data' is a DataFrame containing the stock information
# Merge it with the options chain to have the stock price in the same DataFrame
#merged_data = pd.merge(apple_chain, stock_data, left_on='contractSymbol', right_on='id')

# Apply the function to calculate Delta and Theta for each row
#options_greeks = merged_data.apply(calculate_delta_theta, axis=1)

# Concatenate the results to the original DataFrame
#final_options_data = pd.concat([merged_data, options_greeks], axis=1)


