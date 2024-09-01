import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.dates import DateFormatter

# Parameters
S0 = 100  # Initial stock price
mu = 0.0002  # Expected return per minute
sigma = 0.02  # Increased volatility
minutes = 390  # Trading minutes in a day
dt = 1 / 390  # Fraction of a day per minute

# Simulate intraday stock prices
np.random.seed(42)
price_changes = np.random.normal(mu * dt, sigma * np.sqrt(dt), minutes)
prices = S0 * np.exp(np.cumsum(price_changes))

# Setup time index for trading day
start_time = pd.Timestamp('2023-09-01 09:30:00')
time_index = pd.date_range(start=start_time, periods=minutes, freq='T')
data = pd.DataFrame(prices, index=time_index, columns=['Price'])
data['Short_MA'] = data['Price'].rolling(window=5).mean()
data['Long_MA'] = data['Price'].rolling(window=20).mean()

# Setup the plot for dynamic visualization
fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))  
line1, = ax.plot([], [], 'b-', label='Stock Price')
line2, = ax.plot([], [], 'r-', label='5-Min MA')
line3, = ax.plot([], [], 'g-', label='20-Min MA')
ax.legend()

def update(frame):
    end = min(frame + 1, minutes)
    start = max(0, end - 60)  
    view_window = data.iloc[start:end]

    ax.set_xlim(view_window.index[0], view_window.index[-1])
    ax.set_ylim(view_window['Price'].min() * 0.95, view_window['Price'].max() * 1.05)
    
    line1.set_data(view_window.index, view_window['Price'])
    line2.set_data(view_window.index, view_window['Short_MA'])
    line3.set_data(view_window.index, view_window['Long_MA'])

    return line1, line2, line3

# Animation configuration
frames = np.arange(20, minutes)  
interval_per_frame = 10 * 1000 / (60 / 5)  

ani = FuncAnimation(fig, update, frames=frames, repeat=False, blit=True, interval=interval_per_frame)

plt.show()
