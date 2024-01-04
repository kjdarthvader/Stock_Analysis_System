# Part 1 - Data Collection 
import yfinance as yf

# Define the ticker symbol and time period for data collection
ticker_symbol = "AAPL"  # Example with Apple Inc.
start_date = "2020-01-01"
end_date = "2021-01-01"

# Fetch historical stock data
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Display the first few rows of the data
print(stock_data.head())

# Part 2 - Data Preprocessing
import pandas as pd
import numpy as np

# Check for missing values
missing_values = stock_data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Filling missing values (if any) with the previous values
stock_data.fillna(method='ffill', inplace=True)

# Adding additional useful columns
stock_data['Daily_Return'] = stock_data['Close'].pct_change()  # Daily Return
stock_data['Log_Return'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))  # Logarithmic daily return

# Dropping any remaining NaN values
stock_data.dropna(inplace=True)

# Display the first few rows of the processed data
print(stock_data.head())

# Part 3 - Technical Indicators 
import talib

# Calculate Simple Moving Average (SMA)
stock_data['SMA_20'] = talib.SMA(stock_data['Close'], timeperiod=20)  # 20-day SMA
stock_data['SMA_50'] = talib.SMA(stock_data['Close'], timeperiod=50)  # 50-day SMA

# Calculate Exponential Moving Average (EMA)
stock_data['EMA_20'] = talib.EMA(stock_data['Close'], timeperiod=20)  # 20-day EMA
stock_data['EMA_50'] = talib.EMA(stock_data['Close'], timeperiod=50)  # 50-day EMA

# Calculate Relative Strength Index (RSI)
stock_data['RSI'] = talib.RSI(stock_data['Close'], timeperiod=14)  # 14-day RSI

# Calculate Moving Average Convergence Divergence (MACD)
stock_data['MACD'], stock_data['MACD_signal'], stock_data['MACD_hist'] = talib.MACD(stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

# Display the first few rows with the new indicators
print(stock_data[['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']].head())


