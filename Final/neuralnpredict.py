import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import joblib  # For saving/loading models
import time  # For performance monitoring

def fetch_data(symbol, interval="1m", period="7d"):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period, interval=interval)
    if data.empty:
        raise ValueError("No data found for the given symbol.")
    return data

def prepare_data(stock_data):
    stock_data['Day_Change'] = stock_data['Close'] - stock_data['Open']
    X = stock_data[['Open', 'High', 'Low', 'Volume', 'Day_Change']]
    y = stock_data['Close']
    return X, y

def train_random_forest(X, y):
    model = RandomForestRegressor(n_estimators=50, random_state=42)  # Balancing complexity and speed
    model.fit(X, y)
    joblib.dump(model, 'random_forest_model.pkl')  # Save model for reuse
    return model

def prepare_lstm_data(stock_data):
    data = stock_data['Close'].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def train_lstm(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=10)  # Increase epochs for accuracy
    model.save('lstm_model.h5')  # Save LSTM model
    return model

def predict_lstm_price(model, scaler, recent_data):
    last_60_data = recent_data[-60:].reshape(-1, 1)
    scaled_last_60_data = scaler.transform(last_60_data)
    X_test = np.array([scaled_last_60_data])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_price = model.predict(X_test)
    return scaler.inverse_transform(predicted_price)[0][0]

def evaluate_model(predictions, actuals):
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return mae, rmse

def main():
    symbol = input("Enter the stock symbol (e.g., GOOG): ").upper()
    time_delta = int(input("Enter the prediction time delta in minutes (e.g., 5): "))

    try:
        start_time = time.time()

        # Fetch data
        stock_data = fetch_data(symbol)

        # Prepare data
        X, y = prepare_data(stock_data)
        
        # Train or load Random Forest model
        try:
            rf_model = joblib.load('random_forest_model.pkl')
        except FileNotFoundError:
            rf_model = train_random_forest(X, y)

        # Prepare LSTM data
        X_lstm, y_lstm, scaler = prepare_lstm_data(stock_data)
        
        # Train or load LSTM model
        try:
            lstm_model = keras.models.load_model('lstm_model.h5')
        except IOError:
            lstm_model = train_lstm(X_lstm, y_lstm)

        # Predict future price
        last_minute_data = stock_data['Close'].values
        future_price = predict_lstm_price(lstm_model, scaler, last_minute_data)

        current_time = stock_data.index[-1]
        future_time = current_time + timedelta(minutes=time_delta)

        current_price = stock_data['Close'].iloc[-1]

        print(f"Current Time: {current_time}")
        print(f"Current Price for {symbol}: {current_price:.2f}")
        print(f"Predicted Price for {symbol} at {future_time}: {future_price:.2f}")

        # Evaluate model (if applicable)
        actual_prices = stock_data['Close'].values[-60:]
        predicted_prices = [predict_lstm_price(lstm_model, scaler, last_minute_data) for _ in range(60)]
        
        mae, rmse = evaluate_model(predicted_prices, actual_prices)
        print(f"Model MAE: {mae:.2f}")
        print(f"Model RMSE: {rmse:.2f}")

        print(f"Execution Time: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
