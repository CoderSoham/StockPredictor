"""
This code fetches historical data for Google stock from a CSV file, converts it to a DataFrame, and trains a Random Forest Regression model using the features such as Open, High, Low, Close, and Volume. It then fetches real-time data for Google stock using the yfinance library, continuously updates a plot showing the real-time and predicted stock prices, and predicts the price of the stock at the end of the trading day based on the current situation of the stock.

Features:
- Historical data for Google stock loaded from a CSV file
- Conversion of 'Date' column to datetime format
- Training of Random Forest Regression model with 100 estimators and a random state of 42
- Fetching of real-time data for Google stock using the yfinance library
- Continuous updating of a plot showing real-time and predicted stock prices
- Prediction of the stock price at the end of the trading day based on the current situation
- Display of the predicted end-of-day price in the console

The code runs in an infinite loop, periodically fetching real-time data, predicting the stock price using the trained model, updating the plot, predicting the end-of-day price, and then waiting for 10 seconds before fetching the next data. Any errors that occur during execution are printed to the console.

"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import time

# Load historical data for Google
google_data = pd.read_csv('google.csv')

# Convert 'Date' column to datetime
google_data['Date'] = pd.to_datetime(google_data['Date'])

# Use historical data to train the Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(google_data[['Open', 'High', 'Low', 'Close', 'Volume']], google_data['Adj Close'])

# Fetch real-time data for Google
ticker_symbol = 'GOOGL'

# Create empty lists to store real-time data
google_timestamps = []
google_prices = []
google_predicted_prices = []

# Create the plot
plt.figure(figsize=(12, 6))
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'Real-Time and Predicted Stock Price for {ticker_symbol}')

# Function to update the plot
def update_google_plot():
    plt.plot(google_timestamps, google_prices, label='Real-Time Price', color='green')
    plt.plot(google_timestamps, google_predicted_prices, label='Predicted Price', linestyle='--', color='red')
    plt.legend()
    plt.pause(0.01)

# Function to predict price for the remaining duration of the trading day
def predict_end_of_day_price(today, google_realtime_data):
    trading_hours_end = today.replace(hour=16, minute=0, second=0, microsecond=0)  # Assuming trading hours end at 4:00 PM
    time_remaining = trading_hours_end - datetime.now()
    time_remaining_hours = time_remaining.total_seconds() / 3600
    current_features = [google_realtime_data['Open'].iloc[-1], google_realtime_data['High'].iloc[-1], 
                        google_realtime_data['Low'].iloc[-1], google_realtime_data['Close'].iloc[-1], 
                        google_realtime_data['Volume'].iloc[-1]]
    predicted_price_change = model.predict([current_features])[0] * time_remaining_hours
    return google_latest_price + predicted_price_change

# Infinite loop to continuously update the plot with real-time data
while True:
    try:
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

        # Predict price using the Random Forest Regression model
        google_predicted_price = model.predict([[google_realtime_data['Open'].iloc[-1], google_realtime_data['High'].iloc[-1], 
                                                 google_realtime_data['Low'].iloc[-1], google_realtime_data['Close'].iloc[-1], 
                                                 google_realtime_data['Volume'].iloc[-1]]])[0]
        google_predicted_prices.append(google_predicted_price)

        # Update the plot
        plt.clf()  # Clear the current figure
        update_google_plot()

        # Predict the price at the end of the day
        end_of_day_price = predict_end_of_day_price(today, google_realtime_data)
        print("Predicted price at the end of the day:", end_of_day_price)

        # Wait for 10 seconds before fetching the next data
        time.sleep(10)
    except Exception as e:
        print("Error:", e)
