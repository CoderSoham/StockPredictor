import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import time

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
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'Real-Time and Predicted Stock Price for {symbol}')
    
    # Initialize variables to store previous data
    prev_timestamp = None
    prev_real_price = None
    prev_predicted_price = None
    
    # Infinite loop to continuously update the plot with real-time data
    while True:
        try:
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
            
            # Plot only if there's a change in price
            if latest_price != prev_real_price or predicted_price != prev_predicted_price:
                plt.clf()
                plt.plot([prev_timestamp, latest_timestamp], [prev_real_price, latest_price], label='Real-Time Price', color='green', marker='o')
                plt.plot([prev_timestamp, latest_timestamp], [prev_real_price, predicted_price], label='Predicted Price', linestyle='--', color='red', marker='x')
                plt.legend()
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.title(f'Real-Time and Predicted Stock Price for {symbol}')
                plt.pause(0.01)
                
                # Update previous data
                prev_timestamp = latest_timestamp
                prev_real_price = latest_price
                prev_predicted_price = predicted_price
    
            # Wait for 10 seconds before fetching the next data
            time.sleep(10)
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
