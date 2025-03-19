import yfinance as yf
from datetime import datetime, timedelta
import pytz

# List of stock tickers
stocks = ['RECLTD.NS', 'NTPC.NS', 'COALINDIA.NS', 'PFC.NS', 'TATAPOWER.NS', 'BHEL.NS']

# Get the previous day's date (in IST)
ist = pytz.timezone('Asia/Kolkata')
today_ist = datetime.now(ist).date()
previous_day_ist = today_ist - timedelta(days=1)

# Fetch stock data from Yahoo Finance using yfinance
def get_previous_closing_prices(stocks):
    prices = {}
    for stock in stocks:
        data = yf.Ticker(stock)
        
        # Get historical data for the previous day
        hist = data.history(period='1d', start=previous_day_ist, end=today_ist)
        
        # Fetch the closing price of the previous day
        if not hist.empty:
            closing_price = hist['Close'][-1]  # Get the last close of the previous day
        else:
            closing_price = None  # Handle cases where data might be missing
            
        prices[stock] = closing_price
    return prices

# Get previous closing prices and print them
previous_closing_prices = get_previous_closing_prices(stocks)
for stock, price in previous_closing_prices.items():
    if price:
        print(f"Stock: {stock} | Previous Day Closing Price: {price:.2f} INR")
    else:
        print(f"Stock: {stock} | Data not available")
