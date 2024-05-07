import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import time

# Load historical data for Google
google_data = pd.read_csv('google.csv')

# Convert 'Date' column to datetime
google_data['Date'] = pd.to_datetime(google_data['Date'])

# Use historical data to train the model
model = LinearRegression()
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
    google_predicted_prices.append(google_predicted_price)

    # Update the plot
    plt.clf()  # Clear the current figure
    update_google_plot()

    # Wait for 10 seconds before fetching the next data
    time.sleep(10)
