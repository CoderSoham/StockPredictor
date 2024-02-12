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
predictions = []

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
    plt.plot(timestamps[-1], prices[-1], marker='o', markersize=5, color='green')  # Highlight the latest real-time point
    
    if predicted_prices:
        plt.plot(timestamps[-1], predicted_prices[-1], marker='o', markersize=5, color='red')  # Highlight the predicted point
    
    plt.legend()
    plt.pause(0.01)

# Function to predict next minute price direction
def predict_next_minute():
    # Extract features for the most recent data point
    latest_data_point = data.iloc[-1]
    latest_features = np.array(latest_data_point[['Open', 'High', 'Low', 'Close', 'Volume']]).reshape(1, -1)
    
    # Predict the price for the next minute
    next_minute_price = model.predict(latest_features)[0]
    
    # Store the predicted price
    predicted_prices.append(next_minute_price)
    
    # Determine the price direction (up or down)
    direction = 'Up' if next_minute_price > prices[-1] else 'Down'
    
    # Calculate the percentage chance of the predicted direction
    if direction == 'Up':
        percentage_chance = 100 * (next_minute_price - prices[-1]) / prices[-1]
    else:
        percentage_chance = -100 * (next_minute_price - prices[-1]) / prices[-1]
    
    # Store the prediction
    predictions.append((direction, percentage_chance))

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
    
    # Update the plot
    plt.clf()  # Clear the current figure
    update_plot()
    
    # Predict next minute price direction every 20 seconds
    if len(prices) >= 2 and (timestamps[-1] - timestamps[-2]).seconds >= 20:
        predict_next_minute()
        print(f'Prediction for next minute: {predictions[-1]}')
    
    # Wait for a few seconds before fetching the next data
    time.sleep(1)
