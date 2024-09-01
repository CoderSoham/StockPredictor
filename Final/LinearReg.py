import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta

def fetch_daily_data(symbol):
    stock_data = yf.download(symbol, period="max")
    return stock_data

def fetch_minute_data(symbol):
    stock = yf.Ticker(symbol)
    minute_data = stock.history(period='7d', interval='1m')
    return minute_data

def train_linear_regression_model(stock_data):
    X = stock_data[['Open', 'High', 'Low', 'Volume']]
    y_close = stock_data['Close']
    model_close = LinearRegression()
    model_close.fit(X, y_close)
    return model_close

def predict_linear_regression_future_price(model_close, last_data_point):
    future_data = [[
        last_data_point['Open'].iloc[-1], 
        last_data_point['High'].iloc[-1], 
        last_data_point['Low'].iloc[-1], 
        last_data_point['Volume'].iloc[-1]
    ]]
    future_close = model_close.predict(future_data)[0]
    return future_close

def main():
    symbol = input("Enter the stock symbol (e.g., GOOG): ").upper()
    prediction_minutes = int(input("Enter the prediction time delta in minutes (e.g., 5): "))
    prediction_delta = timedelta(minutes=prediction_minutes)

    stock_data = fetch_daily_data(symbol)
    minute_data = fetch_minute_data(symbol)

    if stock_data.empty or minute_data.empty:
        print("Error: Data not available for the entered stock symbol.")
        return

    model_close_lr = train_linear_regression_model(stock_data)

    # Use the latest available minute data point to predict
    last_data_point = minute_data.tail(1)

    future_price_lr = predict_linear_regression_future_price(model_close_lr, last_data_point)

    current_time = last_data_point.index[-1]
    future_time = current_time + prediction_delta

    current_price = last_data_point['Close'].iloc[-1]

    print(f"Current Time: {current_time}")
    print(f"Current Price for {symbol}: {current_price:.2f}")
    print(f"Predicted Price for {symbol} (LR) at {future_time}: {future_price_lr:.2f}")

if __name__ == "__main__":
    main()
