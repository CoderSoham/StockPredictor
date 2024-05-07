import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from scipy.interpolate import make_interp_spline

def fetch_data(symbol):
    try:
        # Fetch historical data for the stock
        stock_data = yf.download(symbol, period="1mo")  # Fetching data for the last month
        
        # Convert 'Date' column to datetime
        stock_data.reset_index(inplace=True)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        
        return stock_data
    except Exception as e:
        print(f"Error fetching data for symbol {symbol}: {e}")
        return None

def plot_candlestick(stock_data):
    # Sort the stock data by date
    stock_data = stock_data.sort_values(by='Date')
    
    # Perform interpolation
    spline = make_interp_spline(stock_data['Date'], stock_data['Close'], k=3)
    
    # Smooth the data
    x_smooth = pd.date_range(start=stock_data['Date'].min(), end=stock_data['Date'].max(), periods=300)
    y_smooth = spline(x_smooth)
    
    # Plot candlestick chart
    fig, ax = plt.subplots()
    ax.xaxis_date()
    candlestick_ohlc(ax, zip(mdates.date2num(stock_data['Date']), stock_data['Open'], stock_data['High'], stock_data['Low'], stock_data['Close']), width=0.6, colorup='g', colordown='r')
    ax.plot(x_smooth, y_smooth, 'b-', label='Smoothed Close Price')
    ax.set_title('Candlestick Chart for Stock')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    plt.xticks(rotation=45)
    
    plt.show()

def main():
    # Ask user for stock symbol
    symbol = input("Enter the stock symbol (e.g., GOOG): ").upper()
    
    # Fetch data
    stock_data = fetch_data(symbol)
    
    if stock_data is not None:
        # Plot candlestick chart
        plot_candlestick(stock_data)

if __name__ == "__main__":
    main()
