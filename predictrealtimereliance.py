"""
This code fetches historical data for Reliance Industries from a CSV file, converts it to a DataFrame, and trains a linear regression model using features such as Open, High, Low, Close, and Volume. It then fetches real-time data for Reliance Industries using the yfinance library, continuously updates a plot showing the real-time and predicted stock prices, and calculates the moving average based on the real-time data.

Features:
- Historical data for Reliance Industries loaded from a CSV file
- Conversion of 'Date' column to datetime format
- Training of a linear regression model using historical data
- Fetching of real-time data for Reliance Industries using the yfinance library
- Continuous updating of a plot showing real-time and predicted stock prices
- Calculation of the moving average based on the real-time data
- Display of the latest real-time point, predicted point, and moving average point on the plot
- Looping indefinitely to continuously update the plot with real-time data at regular intervals

"""
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import time

# Load historical data
data = pd.read_csv('reliance.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Use historical data to train the model
model = LinearRegression()
model.fit(data[['Open', 'High', 'Low', 'Close', 'Volume']], data['Adj Close'])

# Define the window size for the moving average
window_size = 10

# Fetch real-time data for Reliance Industries
ticker_symbol = 'RELIANCE.NS'

# Create empty lists to store real-time data
timestamps = []
prices = []

# Create the plot
plt.figure(figsize=(10, 6))
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'Real-Time and Predicted Stock Price for {ticker_symbol}')

# Initialize the plot with dummy data
plt.plot([], [], label='Real-Time Price', color='green')
plt.plot([], [], label='Predicted Price', linestyle='--', color='red')
plt.plot([], [], label=f'Moving Average ({window_size} days)', linestyle='-.', color='blue')

plt.legend()
plt.grid(True)

# Function to calculate moving average
def calculate_moving_average(prices):
    if len(prices) >= window_size:
        return sum(prices[-window_size:]) / window_size
    else:
        return None

# Function to update the plot
def update_plot():
    plt.plot(timestamps, prices, label='Real-Time Price', color='green')
    plt.plot(timestamps[-1], prices[-1], marker='o', markersize=5, color='green')  # Highlight the latest real-time point
    plt.plot(timestamps[-1], predicted_price, marker='o', markersize=5, color='red')  # Highlight the predicted point
    
    moving_avg = calculate_moving_average(prices)
    if moving_avg:
        plt.plot(timestamps[-1], moving_avg, marker='o', markersize=5, color='blue')  # Highlight the moving average point
    
    plt.legend()
    plt.pause(0.01)

# Infinite loop to continuously update the plot with real-time data
while True:
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

    # Predict price using the model
    predicted_price = model.predict([[realtime_data['Open'].iloc[-1], realtime_data['High'].iloc[-1], 
                                      realtime_data['Low'].iloc[-1], realtime_data['Close'].iloc[-1], 
                                      realtime_data['Volume'].iloc[-1]]])[0]

    # Update the plot
    plt.clf()  # Clear the current figure
    update_plot()

    # Wait for a few seconds before fetching the next data
    time.sleep(1)
