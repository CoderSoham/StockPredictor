import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf

def fetch_data(symbol):
    # Fetch historical data for the stock
    stock_data = yf.download(symbol, period="max")
    return stock_data

def train_model(stock_data):
    # Split the data into features (X) and target variables (y)
    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y_close = stock_data['Close']
    y_high = stock_data['High']
    y_low = stock_data['Low']

    # Train the Random Forest Regression model
    model_close = RandomForestRegressor(n_estimators=100, random_state=42)
    model_high = RandomForestRegressor(n_estimators=100, random_state=42)
    model_low = RandomForestRegressor(n_estimators=100, random_state=42)

    model_close.fit(X, y_close)
    model_high.fit(X, y_high)
    model_low.fit(X, y_low)

    return model_close, model_high, model_low

def predict_next_day_prices(model_close, model_high, model_low, last_data_point):
    # Predict the next day's closing price, high, and low
    next_day_open = last_data_point['Open'].iloc[-1]
    next_day_volume = last_data_point['Volume'].iloc[-1]

    next_day_data = [[next_day_open, next_day_open, next_day_open, next_day_open, next_day_volume]]

    next_day_close = model_close.predict(next_day_data)[0]
    next_day_high = model_high.predict(next_day_data)[0]
    next_day_low = model_low.predict(next_day_data)[0]

    return next_day_close, next_day_high, next_day_low

def main():
    # Ask user for stock symbol
    symbol = input("Enter the stock symbol (e.g., GOOG): ").upper()

    # Fetch historical data for the chosen stock
    stock_data = fetch_data(symbol)

    # Check if data is available for the entered symbol
    if stock_data.empty:
        print("Error: Data not available for the entered stock symbol.")
        return

    # Train the models
    model_close, model_high, model_low = train_model(stock_data)

    # Get data until the previous day
    last_data_point = stock_data.tail(1)

    # Predict next day's prices
    next_day_close, next_day_high, next_day_low = predict_next_day_prices(model_close, model_high, model_low, last_data_point)

    # Print the predicted prices
    print(f"Predicted Closing Price for {symbol} tomorrow: {next_day_close:.2f}")
    print(f"Predicted High Price for {symbol} tomorrow: {next_day_high:.2f}")
    print(f"Predicted Low Price for {symbol} tomorrow: {next_day_low:.2f}")

if __name__ == "__main__":
    main()
