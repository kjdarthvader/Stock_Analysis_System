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


