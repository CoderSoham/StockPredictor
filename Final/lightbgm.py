import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import time
import os

def fetch_intraday_data(symbol):
    stock = yf.Ticker(symbol)
    stock_data = stock.history(period='7d', interval='1m')
    if stock_data.empty:
        raise ValueError("No data found for the given symbol.")
    return stock_data

def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df.fillna(0, inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_data(stock_data):
    stock_data = add_technical_indicators(stock_data)
    X = stock_data[['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI']]
    y = stock_data['Close']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LGBMRegressor(n_estimators=100, learning_rate=0.1, boosting_type='gbdt')
    model.fit(X_train, y_train)

    joblib.dump(model, 'lightgbm_model.pkl')

    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    return model

def predict_future_price(model, last_data_point):
    features = [
        last_data_point['Open'].iloc[-1],
        last_data_point['High'].iloc[-1],
        last_data_point['Low'].iloc[-1],
        last_data_point['Volume'].iloc[-1],
        last_data_point['SMA_20'].iloc[-1],
        last_data_point['SMA_50'].iloc[-1],
        last_data_point['EMA_12'].iloc[-1],
        last_data_point['EMA_26'].iloc[-1],
        last_data_point['RSI'].iloc[-1]
    ]
    
    future_data = [features]
    future_close = model.predict(future_data)[0]
    return future_close

def main():
    symbol = input("Enter the stock symbol (e.g., GOOG): ").upper()
    prediction_minutes = int(input("Enter the prediction time delta in minutes (e.g., 5): "))
    prediction_delta = timedelta(minutes=prediction_minutes)

    try:
        start_time = time.time()

        # Fetch and prepare data
        stock_data = fetch_intraday_data(symbol)
        X, y = prepare_data(stock_data)

        # Delete the old model if it exists
        if os.path.exists('lightgbm_model.pkl'):
            os.remove('lightgbm_model.pkl')

        # Train and save the model
        model = train_model(X, y)

        # Get the latest data point
        last_data_point = stock_data.tail(1)
        future_price = predict_future_price(model, last_data_point)

        # Calculate times
        current_time = last_data_point.index[-1]
        future_time = current_time + prediction_delta
        current_price = last_data_point['Close'].iloc[-1]

        # Print results
        print(f"Current Time: {current_time}")
        print(f"Current Price for {symbol}: {current_price:.2f}")
        print(f"Predicted Price for {symbol} at {future_time}: {future_price:.2f}")

        print(f"Execution Time: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
