import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
from datetime import datetime, timedelta

def fetch_data(symbol):
    # Fetch historical data for the stock
    stock_data = yf.download(symbol, period="max")
    return stock_data

def train_model(stock_data):
    # Split the data into features (X) and target variables (y)
    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y_close = stock_data['Close']

    # Train the Random Forest Regression model
    model_close = RandomForestRegressor(n_estimators=100, random_state=42)
    model_close.fit(X, y_close)

    return model_close

def predict_30min_price(model_close, last_data_point):
    # Predict the price 30 minutes into the future
    next_30min_data = [[
        last_data_point['Open'].iloc[-1], 
        last_data_point['High'].iloc[-1], 
        last_data_point['Low'].iloc[-1], 
        last_data_point['Close'].iloc[-1], 
        last_data_point['Volume'].iloc[-1]
    ]]
    next_30min_close = model_close.predict(next_30min_data)[0]
    return next_30min_close

def main():
    # Ask user for stock symbol
    symbol = input("Enter the stock symbol (e.g., GOOG): ").upper()

    # Fetch historical data for the chosen stock
    stock_data = fetch_data(symbol)

    # Check if data is available for the entered symbol
    if stock_data.empty:
        print("Error: Data not available for the entered stock symbol.")
        return

    # Train the model
    model_close = train_model(stock_data)

    # Fetch real-time data for the chosen stock
    stock = yf.Ticker(symbol)
    realtime_data = stock.history(period='1d', interval='1m')

    # Get the latest data point
    last_data_point = realtime_data.tail(1)

    # Predict price 60 minutes into the future
    future_price = predict_30min_price(model_close, last_data_point)

    # Get the current time and future time
    current_time = last_data_point.index[-1]
    future_time = current_time + timedelta(minutes=60)

    # Print the predicted price 60 minutes into the future
    print(f"Current Time: {current_time}")
    print(f"Predicted Price for {symbol} at {future_time}: {future_price:.2f}")

if __name__ == "__main__":
    main()
