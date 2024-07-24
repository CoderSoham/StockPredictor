import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import time
import sys

def fetch_data(symbol):
    # Fetch historical data for the stock
    stock_data = yf.download(symbol, period="max")
    return stock_data

def train_model(stock_data):
    # Split the data into features (X) and target variables (y)
    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y_close = stock_data['Close']

    # Train the Random Forest Regression model
    model_close = RandomForestRegressor(n_estimators=100, random_state=42)
    model_close.fit(X, y_close)

    return model_close

def predict_30min_price(model_close, last_data_point):
    # Predict the price 30 minutes into the future
    next_30min_data = [[
        last_data_point['Open'].iloc[-1], 
        last_data_point['High'].iloc[-1], 
        last_data_point['Low'].iloc[-1], 
        last_data_point['Close'].iloc[-1], 
        last_data_point['Volume'].iloc[-1]
    ]]
    next_30min_close = model_close.predict(next_30min_data)[0]
    return next_30min_close

def fetch_data_and_train_model(symbol):
    # Fetch historical data for the stock
    stock_data = yf.download(symbol, period="max")
    
    # Use historical data to train the Random Forest Regression model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']], stock_data['Adj Close'])
    
    return model

def predict_and_plot_realtime_price(model, symbol):
    """
    Predict and plot real-time stock price using the trained model.
    
    Args:
    - model (RandomForestRegressor): Trained Random Forest Regression model
    - symbol (str): Stock symbol
    """
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f'Real-Time and Predicted Stock Price for {symbol}')
    
    # Initialize lists to store data points
    timestamps = []
    real_prices = []
    predicted_prices = []
    error_percentages = []

    try:
        while True:
            # Fetch real-time data
            stock = yf.Ticker(symbol)
            realtime_data = stock.history(period='1d', interval='1m')

            # Extract the latest timestamp and price
            latest_timestamp = realtime_data.index[-1]
            latest_price = realtime_data['Close'].iloc[-1]

            # Predict price for the next 30 minutes
            future_timestamp = latest_timestamp + timedelta(minutes=30)
            next_30min_close = model.predict([[
                realtime_data['Open'].iloc[-1], 
                realtime_data['High'].iloc[-1], 
                realtime_data['Low'].iloc[-1], 
                realtime_data['Close'].iloc[-1], 
                realtime_data['Volume'].iloc[-1]
            ]])[0]

            # Calculate error percentage
            error_percentage = ((latest_price - next_30min_close) / latest_price) * 100

            # Append data to lists
            timestamps.append(latest_timestamp)
            real_prices.append(latest_price)
            predicted_prices.append(next_30min_close)
            error_percentages.append(error_percentage)

            # Clear the plot
            plt.clf()

            # Plot real-time and predicted prices
            plt.plot(timestamps, real_prices, label='Real-Time Price', color='green', marker='o')
            plt.plot(timestamps, predicted_prices, label='Predicted Price', linestyle='--', color='red', marker='x')

            # Add labels and legend
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.title(f'Real-Time and Predicted Stock Price for {symbol} (Error: {error_percentage:.2f}%)')
            plt.legend(loc='best')

            # Set y-axis limits
            plt.ylim(min(min(real_prices), min(predicted_prices)) * 0.95, max(max(real_prices), max(predicted_prices)) * 1.05)

            # Display the plot
            plt.pause(1)

            # Wait for 60 seconds before fetching the next data
            time.sleep(60)
    except KeyboardInterrupt:
        print("Exiting program...")
        sys.exit(0)
    except Exception as e:
        print("Error:", e)

def main():
    # Ask user for stock symbol
    symbol = input("Enter the stock symbol (e.g., GOOG): ").upper()
    
    # Fetch data and train model
    model = fetch_data_and_train_model(symbol)
    
    # Predict and plot real-time price
    predict_and_plot_realtime_price(model, symbol)

if __name__ == "__main__":
    main()
