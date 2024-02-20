import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

start = dt.datetime(2005, 1, 1)
end = dt.datetime(2024, 1, 1)

tickers = ['GOOGL']

data = yf.download(tickers, start=start, end=end)

# Save data to CSV file
data.to_csv('goog.csv')

plt.figure(figsize=(12, 8))

for ticker in tickers:
    plt.plot(data.index, data['Close'][ticker], label=ticker)
    

plt.title('Closing Prices of Multiple Stocks')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(False)
plt.show()
#scrapper code