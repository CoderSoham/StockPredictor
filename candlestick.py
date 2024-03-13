import pandas as pd
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates

# Read the historical stock data from the CSV file
data = pd.read_csv('apple.csv')

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Convert the 'Date' column to matplotlib's internal date format
data['Date'] = data['Date'].apply(mdates.date2num)

# Prepare the data for the candlestick plot
ohlc = data[['Date', 'Open', 'High', 'Low', 'Close']].copy()

# Create a new plot
fig, ax = plt.subplots()

# Plot the candlestick chart
candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='g', colordown='r')

# Format the x-axis to display dates nicely
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

plt.title('Stock Price Candlestick Chart')
plt.xlabel('Date')
plt.ylabel('Price')
plt.tight_layout()
plt.show()
