import yfinance as yf

# Define the ticker symbol and time period for data collection
ticker_symbol = "AAPL"  # Example with Apple Inc.
start_date = "2020-01-01"
end_date = "2021-01-01"

# Fetch historical stock data
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Display the first few rows of the data
print(stock_data.head())
