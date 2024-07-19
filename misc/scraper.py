import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

tickers = ['ORCL']

data = yf.download(tickers, period="max")

data.to_csv('oracle.csv')
