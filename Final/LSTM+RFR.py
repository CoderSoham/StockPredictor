import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def fetch_daily_data(symbol):
    # Fetch only the current year's data
    current_year = datetime.now().year
    stock_data = yf.download(symbol, start=f'{current_year}-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    return stock_data

def fetch_minute_data(symbol):
    stock = yf.Ticker(symbol)
    minute_data = stock.history(period='7d', interval='1m')
    return minute_data

def train_rfr_model(stock_data):
    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y_close = stock_data['Close']
    # Train Random Forest Regressor without a fixed random state
    model_close = RandomForestRegressor(n_estimators=100, random_state=None)
    model_close.fit(X, y_close)
    return model_close

def train_lstm_model(minute_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(minute_data[['Close', 'Volume']])

    X, y = [], []
    time_steps = 60
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 2))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 2)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32)

    return model, scaler, time_steps

def predict_rfr_future_price(model_close, last_data_point):
    future_data = [[
        last_data_point['Open'].iloc[-1], 
        last_data_point['High'].iloc[-1], 
        last_data_point['Low'].iloc[-1], 
        last_data_point['Close'].iloc[-1], 
        last_data_point['Volume'].iloc[-1]
    ]]
    future_close = model_close.predict(future_data)[0]
    return future_close

def predict_lstm_future_price(model, scaler, last_data_point, time_steps):
    last_data_point = last_data_point[['Close', 'Volume']].values[-time_steps:]
    last_data_point = scaler.transform(last_data_point)
    last_data_point = last_data_point.reshape((1, time_steps, 2))

    future_close = model.predict(last_data_point)
    future_close = scaler.inverse_transform([[future_close[0][0], 0]])[0][0]
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

    model_close_rfr = train_rfr_model(stock_data)
    model_close_lstm, scaler, time_steps = train_lstm_model(minute_data)

    last_data_point = minute_data.tail(time_steps)

    future_price_rfr = predict_rfr_future_price(model_close_rfr, last_data_point)
    future_price_lstm = predict_lstm_future_price(model_close_lstm, scaler, last_data_point, time_steps)

    current_time = last_data_point.index[-1]
    future_time = current_time + prediction_delta

    current_price = last_data_point['Close'].iloc[-1]

    print(f"Current Time: {current_time}")
    print(f"Current Price for {symbol}: {current_price:.2f}")
    print(f"Predicted Price for {symbol} (RFR) at {future_time}: {future_price_rfr:.2f}")
    print(f"Predicted Price for {symbol} (LSTM) at {future_time}: {future_price_lstm:.2f}")

if __name__ == "__main__":
    main()
