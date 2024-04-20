import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Define start and end dates
start = dt.datetime(1997, 1, 1)
end = dt.datetime(2024, 1, 1)

# Define ticker symbols (use Yahoo Finance format)
tickers = ['ATGL.NS']

# Download data
data = yf.download(tickers, start=start, end=end)

# Save data to CSV file
data.to_csv('adanigas.csv')

# Plotting
plt.figure(figsize=(12, 8))
for ticker in tickers:
    if 'Close' in data.columns:
        plt.plot(data.index, data['Close'], label=ticker)
    else:
        print(f"Close price data not found for {ticker}")

plt.title('Closing Prices of Multiple Stocks')
plt.xlabel('Date')
plt.ylabel('Price (INR)')  # Assuming the currency is Indian Rupee (INR)
plt.legend()
plt.grid(True)
plt.show()
