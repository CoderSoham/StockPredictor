import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import numpy as np
import matplotlib.pyplot as plt

def fetch_data(symbol):
    try:
        # Fetch historical data for the stock
        stock_data = yf.download(symbol, period="max")
        
        # Convert 'Date' column to datetime
        stock_data.reset_index(inplace=True)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        
        return stock_data
    except Exception as e:
        print(f"Error fetching data for symbol {symbol}: {e}")
        return None

def plot_candlestick(stock_data):
    fig, ax = plt.subplots()
    ax.set_title('Candlestick Chart')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    
    # Plot candlestick chart
    ax.plot(stock_data['Date'], stock_data['Open'], color='black', label='Open')
    ax.plot(stock_data['Date'], stock_data['Close'], color='green', label='Close')
    ax.plot(stock_data['Date'], stock_data['High'], color='red', label='High')
    ax.plot(stock_data['Date'], stock_data['Low'], color='blue', label='Low')
    
    # Plot moving averages
    ax.plot(stock_data['Date'], stock_data['Close'].rolling(window=20).mean(), color='orange', label='20-day Moving Average')
    ax.plot(stock_data['Date'], stock_data['Close'].rolling(window=50).mean(), color='purple', label='50-day Moving Average')
    
    ax.legend()
    plt.xticks(rotation=45)
    plt.show()

def buy_sell(stock_data):
    buy_signals = []
    sell_signals = []
    
    for i in range(1, len(stock_data['Close'])):
        if stock_data['Close'][i] > stock_data['Close'][i-1]:
            buy_signals.append(np.nan)
            sell_signals.append(stock_data['Close'][i])
        elif stock_data['Close'][i] < stock_data['Close'][i-1]:
            buy_signals.append(stock_data['Close'][i])
            sell_signals.append(np.nan)
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
    
    return buy_signals, sell_signals

def main():
    # Ask user for stock symbol
    symbol = input("Enter the stock symbol (e.g., GOOG): ").upper()
    
    # Fetch data
    stock_data = fetch_data(symbol)
    
    if stock_data is not None:
        # Infinite loop to continuously update the plot with real-time data
        try:
            while True:
                # Fetch real-time data
                stock = yf.Ticker(symbol)
                realtime_data = stock.history(period='1d')
                
                # Combine historical data with real-time data
                combined_data = pd.concat([stock_data, realtime_data])
                
                # Plot candlestick chart
                plot_candlestick(combined_data)
                
                # Calculate buy and sell signals
                buy_signals, sell_signals = buy_sell(combined_data)
                
                # Plot buy and sell signals
                plt.plot(combined_data['Date'], buy_signals, marker='^', markersize=10, color='green', label='Buy Signal', linestyle='None')
                plt.plot(combined_data['Date'], sell_signals, marker='v', markersize=10, color='red', label='Sell Signal', linestyle='None')
                
                plt.title('Buy and Sell Signals')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.xticks(rotation=45)
                plt.show()
                
                # Wait for 10 seconds before fetching the next data
                time.sleep(10)
        except KeyboardInterrupt:
            print("Exiting program...")

if __name__ == "__main__":
    main()
