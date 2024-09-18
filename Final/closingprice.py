import yfinance as yf
from datetime import datetime, timedelta
import pytz

# List of stock tickers
stocks = ['RECLTD.NS', 'NTPC.NS', 'COALINDIA.NS', 'PFC.NS', 'TATAPOWER.NS', 'BHEL.NS']

# Get today's date (in IST) and the time of closing in IST (3:30 PM IST)
ist = pytz.timezone('Asia/Kolkata')
today_ist = datetime.now(ist).date()
closing_time = ist.localize(datetime.combine(today_ist, datetime.strptime("15:30", "%H:%M").time()))

# Fetch stock data from Yahoo Finance using yfinance
def get_stock_closing_prices(stocks):
    prices = {}
    for stock in stocks:
        data = yf.Ticker(stock)
        
        # Get historical data
        hist = data.history(period='1d', interval='1m', start=today_ist, end=today_ist + timedelta(days=1))
        
        # Fetch the closing price at 3:30 PM IST
        if closing_time in hist.index:
            closing_price = hist.loc[closing_time]['Close']
        else:
            closing_price = hist['Close'][-1]  # If no exact match, get the last close of the day
            
        prices[stock] = closing_price
    return prices

# Get closing prices and print them
closing_prices = get_stock_closing_prices(stocks)
for stock, price in closing_prices.items():
    print(f"Stock: {stock} | Closing Price: {price:.2f} INR")
