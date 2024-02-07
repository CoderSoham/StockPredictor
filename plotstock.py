import datetime as dt
from matplotlib import pyplot as plt
from matplotlib import style
import yfinance as yf

start = dt.datetime(2015, 1, 1)
end = dt.datetime(2024, 1, 1)

tesla = yf.download('TSLA', start=start, end=end)
apple = yf.download('AAPL', start=start, end=end)

style.use('ggplot')
tesla['Close'].plot(figsize=(8, 8), label='Tesla')
apple['Close'].plot(figsize=(8, 8), label='Apple')
plt.legend()
plt.show()
