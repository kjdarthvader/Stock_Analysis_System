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

# Part 4 - Sentiment Analysis 
import requests
from bs4 import BeautifulSoup

def scrape_yahoo_news(ticker):
    # Yahoo News URL for the stock
    url = f"https://news.yahoo.com/stock/{ticker}"

    # Send a request to the URL
    response = requests.get(url)
    page_content = response.content

    # Parse the content with BeautifulSoup
    soup = BeautifulSoup(page_content, 'html.parser')

    # Find all news articles - Adjust the selector as per the website's structure
    articles = soup.find_all('article')
    
    # Extract and return article headlines and URLs (as an example)
    news_data = []
    for article in articles:
        headline = article.find('h3').get_text()  # Adjust the tag and class based on actual structure
        link = article.find('a')['href']  # Adjust the tag and class based on actual structure
        news_data.append((headline, link))

    return news_data

# Example usage
ticker = 'AAPL'  # Apple Inc. as an example
news_articles = scrape_yahoo_news(ticker)
print(news_articles[:5])  # Print first 5 articles

from textblob import TextBlob

def analyze_sentiment(news_articles):
    sentiment_scores = []
    for headline, link in news_articles:
        analysis = TextBlob(headline)
        sentiment_scores.append(analysis.sentiment.polarity)  # Polarity score
    return sentiment_scores

# Analyze the sentiment of the news headlines
sentiment_scores = analyze_sentiment(news_articles)
print(sentiment_scores)

# Part 5 - RNN

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Data preparation for RNN
def prepare_data(data, n_features):
    data = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X_train = []
    y_train = []

    for i in range(n_features, len(scaled_data)):
        X_train.append(scaled_data[i - n_features:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, scaler

# Number of previous days to use for prediction
n_features = 60
X_train, y_train, scaler = prepare_data(stock_data, n_features)

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Part 6 - Confidence Score Calculation 
def calculate_confidence_score(stock_data, sentiment_scores, rnn_model, scaler, n_features):
    # Get the latest technical indicators
    latest_data = stock_data.iloc[-n_features:]
    latest_close_prices = latest_data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(latest_close_prices)
    X_test = np.array([scaled_data])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # RNN prediction
    predicted_price = rnn_model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    # Get the latest RSI and MACD
    latest_rsi = latest_data['RSI'].iloc[-1]
    latest_macd = latest_data['MACD'].iloc[-1]

    # Normalize these values (example normalization, can be adjusted)
    rsi_score = (latest_rsi / 100.0)  # Assuming RSI ranges from 0 to 100
    macd_score = (latest_macd + 1) / 2  # Assuming MACD ranges from -1 to 1

    # Average sentiment score
    avg_sentiment_score = np.mean(sentiment_scores)

    # Combine scores from RSI, MACD, sentiment, and RNN prediction
    # Adjust weights if necessary
    total_score = (rsi_score + macd_score + avg_sentiment_score + (predicted_price[0][0] / latest_close_prices[-1])) / 4

    return total_score

# Example usage
confidence_score = calculate_confidence_score(stock_data, sentiment_scores, model, scaler, n_features)
print("Confidence Score for the stock:", confidence_score)
