import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# Fetch stock data from Yahoo Finance from IPO to today
def fetch_data(ticker):
    data = yf.download(ticker, start="2010-01-01")
    return data['Close']

# Scale data for neural network performance
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return scaled_data, scaler

# Create dataset for LSTM training
def create_dataset(dataset, look_back=50):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Define LSTM model
def train_lstm(X_train, y_train, epochs=10, batch_size=1):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# Plot predictions, actual stock prices, and deviations
def plot_predictions(actual, predicted, dates, title):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual Price', color='blue')
    plt.plot(dates, predicted, label='Predicted Price', color='green', alpha=0.7)
    plt.fill_between(dates, actual, predicted, where=(predicted > actual), color='red', alpha=0.3, label='Overestimate')
    plt.fill_between(dates, actual, predicted, where=(predicted < actual), color='yellow', alpha=0.3, label='Underestimate')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def main():
    symbol = input("Enter the stock symbol (e.g., GOOG): ").upper()
    data = fetch_data(symbol)
    scaled_data, scaler = scale_data(data)
    X, y = create_dataset(scaled_data)

    # Define training/testing split
    train_size = int(len(X) * 0.80)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates = data.index[50 + 1:]  # Offset by look_back + 1

    # Train model
    lstm_model = train_lstm(X_train.reshape(-1, 50, 1), y_train)

    # Make predictions
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    testPredictLSTM = lstm_model.predict(X_test_reshaped).flatten()

    # Inverse scale predictions
    testPredictLSTM = scaler.inverse_transform(testPredictLSTM.reshape(-1, 1)).flatten()
    testY = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    dates_test = dates[train_size:]

    # Filter for the range 2023 to 2025
    mask = (dates_test >= pd.Timestamp('2023-01-01')) & (dates_test <= pd.Timestamp('2025-12-31'))
    filtered_dates = dates_test[mask]
    filtered_predictions = testPredictLSTM[mask]
    filtered_actual = testY[mask]

    # Plot predictions
    plot_predictions(filtered_actual, filtered_predictions, filtered_dates, 'Stock Price Prediction (2023-2025)')

    # Calculate performance metrics
    mse = mean_squared_error(filtered_actual, filtered_predictions)
    mae = mean_absolute_error(filtered_actual, filtered_predictions)
    r2 = r2_score(filtered_actual, filtered_predictions)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("Coefficient of Determination (R²):", r2)

if __name__ == '__main__':
    main()
