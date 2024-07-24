import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import time
import sys

def fetch_data_and_train_model(symbol):
    # Fetch historical data for the stock
    stock_data = yf.download(symbol, period="max")
    
    # Convert 'Date' column to datetime
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    
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
    
    # Initialize variables to store previous data
    prev_timestamp = None
    prev_real_price = None
    prev_predicted_price = None
    
    # Initialize lists to store data points
    timestamps = []
    real_prices = []
    predicted_prices = []
    error_percentages = []
    
    # Infinite loop to continuously update the plot with real-time data
    try:
        while True:
            # Fetch real-time data
            stock = yf.Ticker(symbol)
            realtime_data = stock.history(period='1d')
    
            # Extract the latest timestamp and price
            latest_timestamp = datetime.now()
            latest_price = realtime_data['Close'].iloc[-1]
    
            # Predict price using the Random Forest Regression model
            predicted_price = model.predict([[
                realtime_data['Open'].iloc[-1], 
                realtime_data['High'].iloc[-1], 
                realtime_data['Low'].iloc[-1], 
                realtime_data['Close'].iloc[-1], 
                realtime_data['Volume'].iloc[-1]
            ]])[0]
            
            # Calculate error percentage
            error_percentage = ((latest_price - predicted_price) / latest_price) * 100
            
            # Append data to lists
            timestamps.append(latest_timestamp)
            real_prices.append(latest_price)
            predicted_prices.append(predicted_price)
            error_percentages.append(error_percentage)
            
            # Plot the real-time and predicted prices
            plt.plot(timestamps, real_prices, label='Real-Time Price', color='green', marker='o')
            
            # Create a copy of predicted prices list to add 10-second delay
            predicted_prices_copy = predicted_prices.copy()
            # Add predicted price for the next time step (10 seconds ahead)
            predicted_prices_copy.append(predicted_price)
            plt.plot(timestamps + [latest_timestamp + pd.Timedelta(seconds=10)], predicted_prices_copy, label='Predicted Price', linestyle='--', color='red', marker='x')
            
            plt.title(f'Real-Time and Predicted Stock Price for {symbol} (Error: {error_percentage:.2f}%)')
            plt.legend().remove()  # Remove legend
            
            # Set y-axis limits for a smaller scale
            plt.ylim(min(min(real_prices), min(predicted_prices)) * 0.95, max(max(real_prices), max(predicted_prices)) * 1.05)
            
            plt.pause(0.01)
                
            # Update previous data
            prev_timestamp = latest_timestamp
            prev_real_price = latest_price
            prev_predicted_price = predicted_price
    
            # Wait for 30 seconds before fetching the next data
            time.sleep(30)
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
