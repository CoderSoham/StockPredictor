import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
from datetime import datetime, timedelta

def fetch_data(symbol, interval, period='max'):
    # Fetch historical data for the stock with the given interval
    stock_data = yf.download(symbol, interval=interval, period=period)
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
        last_data_point['Open'], 
        last_data_point['High'], 
        last_data_point['Low'], 
        last_data_point['Close'], 
        last_data_point['Volume']
    ]]
    future_close = model_close.predict(future_data)[0]
    return future_close

def main():
    stocks_1h = ["RECLTD.BO", "NTPC.BO", "BHEL.BO"]
    stocks_5h = ["SBIN.BO", "PNB.BO"]
    prediction_delta = timedelta(minutes=60)  # Predict 1 hour ahead for all stocks

    for symbol in stocks_1h + stocks_5h:
        interval = '1h' if symbol in stocks_1h else '5h'
        stock_data = fetch_data(symbol, interval=interval, period="1mo")

        # Check if data is available for the entered symbol
        if stock_data.empty:
            print(f"Error: Data not available for the stock symbol {symbol}.")
            continue

        # Train the model
        model_close = train_model(stock_data)

        # Get the latest data point
        last_data_point = stock_data.iloc[-1]

        # Predict price into the future
        future_price = predict_future_price(model_close, last_data_point)

        # Get the current time and future time
        current_time = last_data_point.name
        future_time = current_time + prediction_delta

        # Get the current price
        current_price = last_data_point['Close']

        # Print the current price and predicted price into the future
        print(f"Current Time: {current_time}")
        print(f"Current Price for {symbol}: {current_price:.2f}")
        print(f"Predicted Price for {symbol} at {future_time}: {future_price:.2f}")

if __name__ == "__main__":
    main()
