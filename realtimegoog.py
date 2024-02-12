import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import time
import numpy as np

# Load historical data for Google
google_data = pd.read_csv('google.csv')

# Convert 'Date' column to datetime
google_data['Date'] = pd.to_datetime(google_data['Date'])

# Use historical data to train the model
model = LinearRegression()
model.fit(google_data[['Open', 'High', 'Low', 'Close', 'Volume']], google_data['Adj Close'])

# Define the window size for the moving average
window_size = 10

# Define the threshold for considering a movement as 'UP'
threshold = 0.1  # Adjust this threshold as needed

# Fetch real-time data for Google
ticker_symbol = 'GOOGL'

# Create empty lists to store real-time data
google_timestamps = []
google_prices = []
google_predicted_prices = []
google_moving_averages = []
predictions = []

# Create the plot
plt.figure(figsize=(12, 6))
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'Real-Time and Predicted Stock Price for {ticker_symbol}')

# Function to calculate moving average
def calculate_moving_average(prices):
    if len(prices) >= window_size:
        return sum(prices[-window_size:]) / window_size
    else:
        return None

# Function to update the plot
def update_google_plot():
    plt.plot(google_timestamps, google_prices, label='Real-Time Price', color='green')
    plt.plot(google_timestamps, google_predicted_prices, label='Predicted Price', linestyle='--', color='red')
    plt.plot(google_timestamps, google_moving_averages, label=f'Moving Average ({window_size} days)', linestyle='-.', color='blue')
    for idx, pred in enumerate(predictions):
        plt.text(google_timestamps[idx], google_prices[idx], pred, ha='right', va='bottom', color='blue' if pred == 'UP' else 'red')
    plt.legend()
    plt.pause(0.01)

# Infinite loop to continuously update the plot with real-time data
while True:
    # Fetch real-time data
    today = datetime.now().date()
    google_stock = yf.Ticker(ticker_symbol)
    google_realtime_data = google_stock.history(period='1d')

    # Extract the latest timestamp and price
    google_latest_timestamp = datetime.now()
    google_latest_price = google_realtime_data['Close'].iloc[-1]

    # Append data to lists
    google_timestamps.append(google_latest_timestamp)
    google_prices.append(google_latest_price)

    # Predict price using the model
    google_predicted_price = model.predict([[google_realtime_data['Open'].iloc[-1], google_realtime_data['High'].iloc[-1], 
                                             google_realtime_data['Low'].iloc[-1], google_realtime_data['Close'].iloc[-1], 
                                             google_realtime_data['Volume'].iloc[-1]]])[0]
    google_predicted_price += np.random.normal(scale=0.1)  # Add some random noise to make the prediction slightly different
    google_predicted_prices.append(google_predicted_price)

    # Calculate moving average
    google_moving_average = calculate_moving_average(google_prices)
    google_moving_averages.append(google_moving_average)

    # Predicting the stock movement (up or down) in the next minute
    current_features = [google_realtime_data['Open'].iloc[-1], google_realtime_data['High'].iloc[-1], 
                        google_realtime_data['Low'].iloc[-1], google_realtime_data['Close'].iloc[-1], 
                        google_realtime_data['Volume'].iloc[-1]]
    predicted_price_change = model.predict([current_features])[0] - google_latest_price
    if predicted_price_change > threshold:
        predictions.append('UP')
    else:
        predictions.append('DOWN')

    # Update the plot
    plt.clf()  # Clear the current figure
    update_google_plot()

    # Wait for 10 seconds before fetching the next data
    time.sleep(10)
