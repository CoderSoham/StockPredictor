import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Function to fetch daily stock data
def fetch_daily_data(symbol):
    current_year = datetime.now().year
    stock_data = yf.download(symbol, start=f'{current_year}-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    return stock_data

# Function to fetch minute-level stock data for 7 days
def fetch_minute_data(symbol):
    stock = yf.Ticker(symbol)
    minute_data = stock.history(period='7d', interval='1m')
    return minute_data

# Random Forest Regressor training function
def train_rfr_model(stock_data):
    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y_close = stock_data['Close']
    model_close = RandomForestRegressor(n_estimators=100, random_state=None)
    model_close.fit(X, y_close)
    return model_close

# LSTM model training function
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

# Random Forest Regressor prediction function
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

# LSTM model prediction function
def predict_lstm_future_price(model, scaler, last_data_point, time_steps):
    last_data_point = last_data_point[['Close', 'Volume']].values[-time_steps:]
    last_data_point = scaler.transform(last_data_point)
    last_data_point = last_data_point.reshape((1, time_steps, 2))

    future_close = model.predict(last_data_point)
    future_close = scaler.inverse_transform([[future_close[0][0], 0]])[0][0]
    return future_close

# Function to predict the price 3 times and find the range
def run_predictions_for_stock(symbol, price_range):
    stock_data = fetch_daily_data(symbol)
    minute_data = fetch_minute_data(symbol)
    
    if stock_data.empty or minute_data.empty:
        print(f"Error: Data not available for {symbol}.")
        return
    
    # Train models
    model_close_rfr = train_rfr_model(stock_data)
    model_close_lstm, scaler, time_steps = train_lstm_model(minute_data)

    last_data_point = minute_data.tail(time_steps)
    
    # Run predictions 3 times
    rfr_predictions = []
    lstm_predictions = []
    
    for _ in range(3):
        future_price_rfr = predict_rfr_future_price(model_close_rfr, last_data_point)
        future_price_lstm = predict_lstm_future_price(model_close_lstm, scaler, last_data_point, time_steps)
        
        rfr_predictions.append(future_price_rfr)
        lstm_predictions.append(future_price_lstm)
    
    # Print results
    min_price, max_price = price_range
    print(f"\nPredictions for {symbol} (Target Range: {min_price}-{max_price} INR):")
    print(f"RFR Predictions: {rfr_predictions}")
    print(f"LSTM Predictions: {lstm_predictions}")
    
    in_range_rfr = [price for price in rfr_predictions if min_price <= price <= max_price]
    in_range_lstm = [price for price in lstm_predictions if min_price <= price <= max_price]
    
    print(f"RFR Prices in Range: {in_range_rfr}")
    print(f"LSTM Prices in Range: {in_range_lstm}")

def main():
    stock_ranges = {
        'RECLTD.NS': (547, 549),
        'NTPC.NS': (423, 426),
        'COALINDIA.NS': (496, 498),
        'PFC.NS': (490, 493),
        'TATAPOWER.NS': (453, 456),
        'BHEL.NS': (269, 272)
    }
    
    for symbol, price_range in stock_ranges.items():
        run_predictions_for_stock(symbol, price_range)

if __name__ == "__main__":
    main()
