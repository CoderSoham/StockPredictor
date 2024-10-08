import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
from datetime import datetime, timedelta

def fetch_data(symbol):
    
    stock_data = yf.download(symbol, period="max")
    return stock_data

def train_model(stock_data):
    
    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y_close = stock_data['Close']

    
    model_close = RandomForestRegressor(n_estimators=100, random_state=42)
    model_close.fit(X, y_close)

    return model_close

def predict_future_price(model_close, last_data_point, prediction_delta):
    
    future_data = [[
        last_data_point['Open'].iloc[-1], 
        last_data_point['High'].iloc[-1], 
        last_data_point['Low'].iloc[-1], 
        last_data_point['Close'].iloc[-1], 
        last_data_point['Volume'].iloc[-1]
    ]]
    future_close = model_close.predict(future_data)[0]
    return future_close

def main():
    
    symbol = input("Enter the stock symbol (e.g., GOOG): ").upper()

    
    prediction_minutes = int(input("Enter the prediction time delta in minutes (e.g., 5): "))
    prediction_delta = timedelta(minutes=prediction_minutes)

    
    stock_data = fetch_data(symbol)

    
    if stock_data.empty:
        print("Error: Data not available for the entered stock symbol.")
        return

    
    model_close = train_model(stock_data)

    
    stock = yf.Ticker(symbol)
    realtime_data = stock.history(period='1d', interval='1m')

    
    last_data_point = realtime_data.tail(1)

    
    future_price = predict_future_price(model_close, last_data_point, prediction_delta)

    
    current_time = last_data_point.index[-1]
    future_time = current_time + prediction_delta

    
    current_price = last_data_point['Close'].iloc[-1]

    
    print(f"Current Time: {current_time}")
    print(f"Current Price for {symbol}: {current_price:.2f}")
    print(f"Predicted Price for {symbol} at {future_time}: {future_price:.2f}")

if __name__ == "__main__":
    main()
