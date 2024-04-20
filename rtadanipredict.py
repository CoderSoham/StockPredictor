import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import time

# Load historical data for Google
google_data = pd.read_csv('adanigas.csv')

# Convert 'Date' column to datetime
google_data['Date'] = pd.to_datetime(google_data['Date'])

# Use historical data to train the Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(google_data[['Open', 'High', 'Low', 'Close', 'Volume']], google_data['Adj Close'])

# Fetch real-time data for Google
ticker_symbol = 'ATGL.NS'

# Create empty lists to store real-time data
google_timestamps = []
google_prices = []
google_predicted_prices = []

# Create the plot
plt.figure(figsize=(12, 6))
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'Real-Time and Predicted Stock Price for {ticker_symbol}')
plt.ion()  # Turn on interactive mode

# Function to calculate moving average
def calculate_moving_average(prices):
    return sum(prices) / len(prices) if prices else None

# Function to update the plot
def update_google_plot():
    plt.gca().cla()  # Clear the current plot axes
    plt.plot(google_timestamps, google_prices, label='Real-Time Price', color='green')
    plt.plot(google_timestamps, google_predicted_prices, label='Predicted Price', linestyle='--', color='red')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'Real-Time and Predicted Stock Price for {ticker_symbol}\nPercentage Error in Predicted Value: {percentage_error:.2f}% \nEnd of Day: {end_of_day_price:.2f}, End of Week: {end_of_week_price:.2f}, End of Month: {end_of_month_price:.2f}')
    plt.pause(0.01)

# Function to predict end price for the day
def predict_end_of_day_price(today, realtime_data):
    # Get historical data for the same day of the week
    historical_data_same_weekday = google_data[google_data['Date'].dt.weekday == today.weekday()]
    average_price_change_same_day = historical_data_same_weekday['Close'].diff().mean()
    return realtime_data['Close'].iloc[-1] + average_price_change_same_day

# Function to predict end price for the week
def predict_end_of_week_price(today, realtime_data):
    # Get historical data for the same weekday over several weeks
    historical_data_same_weekday = google_data[google_data['Date'].dt.weekday == today.weekday()]
    average_price_change_same_week = historical_data_same_weekday['Close'].diff(periods=5).mean()  # Change 5 to desired number of weeks
    return realtime_data['Close'].iloc[-1] + average_price_change_same_week

# Function to predict end price for the month
def predict_end_of_month_price(today, realtime_data):
    # Get historical data for the same month over several years
    historical_data_same_month = google_data[google_data['Date'].dt.month == today.month]
    average_price_change_same_month = historical_data_same_month['Close'].diff(periods=12).mean()  # Change 12 to desired number of months
    return realtime_data['Close'].iloc[-1] + average_price_change_same_month

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

        # Predict end-of-day price
        end_of_day_price = predict_end_of_day_price(today, google_realtime_data)

        # Predict end-of-week price
        end_of_week_price = predict_end_of_week_price(today, google_realtime_data)

        # Predict end-of-month price
        end_of_month_price = predict_end_of_month_price(today, google_realtime_data)

        # Update the plot
        update_google_plot()

        # Wait for 10 seconds before fetching the next data
        time.sleep(10)
    except Exception as e:
        print("Error:", e)
