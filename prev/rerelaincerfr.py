"""

"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import time
import numpy as np

# Load historical data
data = pd.read_csv('reliance.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Use historical data to train the Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(data[['Open', 'High', 'Low', 'Close', 'Volume']], data['Adj Close'])

# Define the window size for the moving average
window_size = 10

# Fetch real-time data for Reliance Industries
ticker_symbol = 'RELIANCE.NS'

# Create empty lists to store real-time data
timestamps = []
prices = []
predicted_prices = []

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
def update_plot():
    plt.plot(timestamps, prices, label='Real-Time Price', color='green')
    plt.plot(timestamps, predicted_prices, label='Predicted Price', linestyle='--', color='red')
    plt.legend()
    plt.pause(0.01)

# Function to predict end-of-day price
def predict_end_of_day_price(today, realtime_data):
    # Assuming trading hours end at 4:00 PM
    trading_hours_end = today.replace(hour=16, minute=0, second=0, microsecond=0)
    time_remaining = trading_hours_end - datetime.now()
    time_remaining_hours = time_remaining.total_seconds() / 3600
    current_features = [realtime_data['Open'].iloc[-1], realtime_data['High'].iloc[-1], 
                        realtime_data['Low'].iloc[-1], realtime_data['Close'].iloc[-1], 
                        realtime_data['Volume'].iloc[-1]]
    predicted_price_change = model.predict([current_features])[0] * time_remaining_hours
    return prices[-1] + predicted_price_change

# Infinite loop to continuously update the plot with real-time data
while True:
    try:
        # Fetch real-time data
        today = datetime.now().date()
        reliance_stock = yf.Ticker(ticker_symbol)
        realtime_data = reliance_stock.history(period='1d')

        # Extract the latest timestamp and price
        latest_timestamp = datetime.now()
        latest_price = realtime_data['Close'].iloc[-1]

        # Append data to lists
        timestamps.append(latest_timestamp)
        prices.append(latest_price)

        # Predict price using the Random Forest Regression model
        predicted_price = model.predict([[realtime_data['Open'].iloc[-1], realtime_data['High'].iloc[-1], 
                                           realtime_data['Low'].iloc[-1], realtime_data['Close'].iloc[-1], 
                                           realtime_data['Volume'].iloc[-1]]])[0]
        predicted_prices.append(predicted_price)

        # Update the plot
        plt.clf()  # Clear the current figure
        update_plot()

        # Predict end-of-day price
        end_of_day_price = predict_end_of_day_price(today, realtime_data)
        print("Predicted price at the end of the day:", end_of_day_price)

        # Wait for 10 seconds before fetching the next data
        time.sleep(10)
    except Exception as e:
        print("Error:", e)
