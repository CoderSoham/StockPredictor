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

plt.legend()
plt.grid(True)

# Function to update the plot
def update_plot():
    plt.plot(timestamps, prices, label='Real-Time Price', color='green')
    plt.plot(timestamps[-1], prices[-1], marker='o', markersize=5, color='green')  # Highlight the latest real-time point
    plt.plot(timestamps[-1], predicted_price, marker='o', markersize=5, color='red')  # Highlight the predicted point
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
