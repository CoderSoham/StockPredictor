import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Additional Imports for Technical Indicators
import talib

def fetch_data(symbol, start_date):
    """Fetch historical stock data."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def compute_technical_indicators(df):
    """Compute technical indicators for the DataFrame."""
    df['MA_50'] = talib.MA(df['Close'], timeperiod=50)
    df['MA_200'] = talib.MA(df['Close'], timeperiod=200)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'])
    df['RSI'] = talib.RSI(df['Close'])
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = talib.BBANDS(df['Close'])
    return df.dropna()

def prepare_data(df, feature_cols):
    """Prepare data for LSTM, including scaling."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_cols])

    # Prepare data for LSTM
    X, y = [], []
    time_steps = 60  # Use 60 timesteps in the past to predict the next step
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i])
        y.append(scaled_data[i, df.columns.get_loc('Close')])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_lstm_model(input_shape):
    """Build LSTM Model."""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    symbol = 'AAPL'  # Example symbol
    start_date = '2010-01-01'  # More historical data

    data = fetch_data(symbol, start_date)
    data_with_indicators = compute_technical_indicators(data)
    feature_cols = ['MA_50', 'MA_200', 'MACD', 'RSI', 'ATR', 'Bollinger_Upper', 'Bollinger_Lower', 'Volume', 'Close']
    X, y, scaler = prepare_data(data_with_indicators, feature_cols)
    
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=10, batch_size=32)
    print("Model trained successfully.")

if __name__ == "__main__":
    main()
