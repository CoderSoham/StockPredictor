import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

def fetch_data(symbol):
    stock_data = yf.download(symbol, period="1y", interval="1d")
    if stock_data.empty:
        raise ValueError("No data found for the given symbol.")
    return stock_data

def fetch_minute_data(symbol):
    stock = yf.Ticker(symbol)
    minute_data = stock.history(period="7d", interval="1m")
    if minute_data.empty:
        raise ValueError("No minute-level data found for the given symbol.")
    return minute_data

def prepare_data_for_random_forest(stock_data):
    # Use daily data features
    stock_data['Day_Change'] = stock_data['Close'] - stock_data['Open']
    X = stock_data[['Open', 'High', 'Low', 'Volume', 'Day_Change']]
    y_close = stock_data['Close']
    return X, y_close

def train_random_forest(X, y_close):
    model_close = RandomForestRegressor(n_estimators=100, random_state=42)
    model_close.fit(X, y_close)
    return model_close

def prepare_data_for_lstm(stock_data):
    # Use minute-level data for LSTM
    data = stock_data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X_train = []
    y_train = []

    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, scaler

def train_lstm(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=10)  # Increased epochs for better training

    return model

def predict_future_price(model, scaler, recent_data):
    last_60_data = recent_data[-60:].reshape(-1, 1)
    scaled_last_60_data = scaler.transform(last_60_data)
    X_test = [scaled_last_60_data]
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][0]

def evaluate_model(predictions, actuals):
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return mae, rmse

def main():
    symbol = input("Enter the stock symbol (e.g., GOOG): ").upper()
    time_delta = int(input("Enter the prediction time delta in minutes (e.g., 5): "))

    stock_data_daily = fetch_data(symbol)
    stock_data_minute = fetch_minute_data(symbol)

    X_daily, y_daily = prepare_data_for_random_forest(stock_data_daily)
    rf_model = train_random_forest(X_daily, y_daily)

    X_minute, y_minute, scaler = prepare_data_for_lstm(stock_data_minute)
    lstm_model = train_lstm(X_minute, y_minute)

    last_minute_data = stock_data_minute['Close'].values
    future_price = predict_future_price(lstm_model, scaler, last_minute_data)

    current_time = stock_data_minute.index[-1]
    future_time = current_time + timedelta(minutes=time_delta)

    current_price = stock_data_minute['Close'].iloc[-1]

    print(f"Current Time: {current_time}")
    print(f"Current Price for {symbol}: {current_price:.2f}")
    print(f"Predicted Price for {symbol} at {future_time}: {future_price:.2f}")

    actual_prices = stock_data_minute['Close'].values[-60:]
    predicted_prices = [predict_future_price(lstm_model, scaler, last_minute_data) for _ in range(60)]
    
    mae, rmse = evaluate_model(predicted_prices, actual_prices)
    print(f"Model MAE: {mae:.2f}")
    print(f"Model RMSE: {rmse:.2f}")

if __name__ == "__main__":
    main()
