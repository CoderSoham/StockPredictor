import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Define the ticker symbol
ticker_symbol = 'GOOGL'  # 'GOOGL' for Google

# Initialize lists to store data
timestamps = []
prices = []

# Create the plot
plt.figure(figsize=(10, 6))
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'Real-Time Stock Price for {ticker_symbol}')

# Initialize the plot with dummy data
plt.plot([], [], label='Price', color='blue')

plt.legend()
plt.grid(True)

# Function to update the plot
def update_plot():
    plt.plot(timestamps, prices, label='Price', color='blue')
    plt.legend()
    plt.pause(0.01)

# Infinite loop to continuously update the plot with real-time data
while True:
    # Fetch the real-time data
    google_stock = yf.Ticker(ticker_symbol)
    realtime_data = google_stock.history(period='1d')
    
    # Extract the latest timestamp and price
    latest_timestamp = datetime.now()
    latest_price = realtime_data['Close'].iloc[-1]
    
    # Append data to lists
    timestamps.append(latest_timestamp)
    prices.append(latest_price)
    
    # Update the plot
    plt.clf()  # Clear the current figure
    update_plot()
    
    # Wait for a few seconds before fetching the next data
    time.sleep(1)
