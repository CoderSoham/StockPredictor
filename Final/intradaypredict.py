import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
from datetime import datetime, timedelta

def fetch_intraday_data(symbol):
    # Fetch minute-by-minute data for the current day
    stock = yf.Ticker(symbol)
    stock_data = stock.history(period='1d', interval='1m')
    return stock_data

def train_model(stock_data):
    # Split the data into features (X) and target variables (y)
    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y_close = stock_data['Close']

    # Train the Random Forest Regression model
    model_close = RandomForestRegressor(n_estimators=100, random_state=42)
    model_close.fit(X, y_close)

    return model_close

def predict_future_price(model_close, last_data_point):
    # Predict the price into the future
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
    # Ask user for stock symbol
    symbol = input("Enter the stock symbol (e.g., GOOG): ").upper()

    # Ask user for the prediction time delta in minutes
    prediction_minutes = int(input("Enter the prediction time delta in minutes (e.g., 5): "))
    prediction_delta = timedelta(minutes=prediction_minutes)

    # Fetch intraday data for the chosen stock
    stock_data = fetch_intraday_data(symbol)

    # Check if data is available for the entered symbol
    if stock_data.empty:
        print("Error: Data not available for the entered stock symbol.")
        return

    # Train the model
    model_close = train_model(stock_data)

    # Get the latest data point
    last_data_point = stock_data.tail(1)

    # Predict price into the future
    future_price = predict_future_price(model_close, last_data_point)

    # Get the current time and future time
    current_time = last_data_point.index[-1]
    future_time = current_time + prediction_delta

    # Get the current price
    current_price = last_data_point['Close'].iloc[-1]

    # Print the current price and predicted price into the future
    print(f"Current Time: {current_time}")
    print(f"Current Price for {symbol}: {current_price:.2f}")
    print(f"Predicted Price for {symbol} at {future_time}: {future_price:.2f}")

if __name__ == "__main__":
    main()
