import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import time
import numpy as np

# Load historical data
data = pd.read_csv('reliance.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Use historical data to train the model
model = LinearRegression()
model.fit(data[['Open', 'High', 'Low', 'Close', 'Volume']], data['Adj Close'])

# Fetch real-time data for Reliance Industries
ticker_symbol = 'RELIANCE.NS'

# Create empty lists to store real-time data
timestamps = []
prices = []
predicted_prices = []
errors = []

# Create the plot
plt.figure(figsize=(10, 6))
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'Real-Time Stock Price and Predictions for {ticker_symbol}')

# Initialize the plot with dummy data
plt.plot([], [], label='Real-Time Price', color='green')
plt.plot([], [], label='Predicted Price', linestyle='--', color='red')

plt.legend()
plt.grid(True)

# Function to update the plot
def update_plot():
    plt.plot(timestamps, prices, label='Real-Time Price', color='green')
    plt.plot(timestamps, predicted_prices, label='Predicted Price', linestyle='--', color='red')
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
    
    # Predict the price for the next minute
    latest_features = np.array(realtime_data[['Open', 'High', 'Low', 'Close', 'Volume']]).reshape(1, -1)
    next_minute_price = model.predict(latest_features)[0]
    predicted_prices.append(next_minute_price)
    
    # Calculate the error percentage between real-time and predicted data
    error_percentage = 100 * (next_minute_price - latest_price) / latest_price
    errors.append(error_percentage)
    
    # Update the plot
    plt.clf()  # Clear the current figure
    update_plot()
    
    # Print the error percentage
    print(f'Error Percentage: {error_percentage:.2f}%')
    
    # Wait for a few seconds before fetching the next data
    time.sleep(5)
