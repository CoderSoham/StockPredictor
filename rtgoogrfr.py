"""
This code fetches historical data for Google stock from a CSV file, converts it to a DataFrame, and trains a Random Forest Regression model using the features such as Open, High, Low, Close, and Volume. It then fetches real-time data for Google stock using the yfinance library, and continuously updates a plot showing the real-time and predicted stock prices.

Features:
- Historical data for Google stock loaded from a CSV file
- Conversion of 'Date' column to datetime format
- Training of Random Forest Regression model with 100 estimators and a random state of 42
- Fetching of real-time data for Google stock using the yfinance library
- Continuous updating of a plot showing real-time and predicted stock prices
- Calculation of percentage error in the predicted value compared to the actual value
- Display of the percentage error in the plot title

The code runs in an infinite loop, periodically fetching real-time data, predicting the stock price using the trained model, updating the plot, and then waiting for 10 seconds before fetching the next data. Any errors that occur during execution are printed to the console.

"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
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
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'Real-Time and Predicted Stock Price for {ticker_symbol}\nPercentage Error in Predicted Value: {percentage_error:.2f}%')
    plt.pause(0.01)

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

        # Calculate the percentage error in predicted value
        actual_price = google_realtime_data['Close'].iloc[-1]
        percentage_error = abs(google_predicted_price - actual_price) / actual_price * 100

        # Update the plot
        plt.clf()  # Clear the current figure
        update_google_plot()

        # Wait for 10 seconds before fetching the next data
        time.sleep(10)
    except Exception as e:
        print("Error:", e)
