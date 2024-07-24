import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fetch_data(ticker):
    start_date = '2004-01-01'
    end_date = '2024-12-31'
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def prepare_signals(stock_data):
    signal_data = pd.DataFrame(index=stock_data.index)
    signal_data['price'] = stock_data['Adj Close']
    signal_data['daily_difference'] = signal_data['price'].diff()
    signal_data['signal'] = np.where(signal_data['daily_difference'] > 0, 1.0, 0.0)
    signal_data['positions'] = signal_data['signal'].diff()
    return signal_data

def plot_signals(signal_data, timeframe):
    fig, ax1 = plt.subplots(figsize=(12, 6), ylabel=f'{ticker} price in $')
    ax1.set_title(f'{ticker} Price and Buy/Sell Signals for {timeframe}')
    signal_data['price'].plot(ax=ax1, color='r', lw=2.)

    ax1.plot(signal_data.loc[signal_data.positions == 1.0].index,
             signal_data.price[signal_data.positions == 1.0],
             '^', markersize=5, color='m', label='Buy')

    ax1.plot(signal_data.loc[signal_data.positions == -1.0].index,
             signal_data.price[signal_data.positions == -1.0],
             'v', markersize=5, color='k', label='Sell')

    ax1.legend()
    plt.show()

def calculate_portfolio(signal_data, initial_capital=1000.0):
    positions = pd.DataFrame(index=signal_data.index).fillna(0.0)
    portfolio = pd.DataFrame(index=signal_data.index).fillna(0.0)
    positions[ticker] = signal_data['signal']
    portfolio['positions'] = (positions.multiply(signal_data['price'], axis=0))
    portfolio['cash'] = initial_capital - (positions.diff().multiply(signal_data['price'], axis=0)).cumsum()
    portfolio['total'] = portfolio['positions'] + portfolio['cash']
    return portfolio

def plot_portfolio(portfolio, signal_data):
    fig, ax1 = plt.subplots(figsize=(12, 6), ylabel='Portfolio value in $')
    ax1.set_title(f'Portfolio Value with Buy/Sell Signals for {ticker}')
    portfolio['total'].plot(ax=ax1, lw=2.)

    ax1.plot(portfolio.loc[signal_data.positions == 1.0].index,
             portfolio.total[signal_data.positions == 1.0],
             '^', markersize=10, color='m', label='Buy')

    ax1.plot(portfolio.loc[signal_data.positions == -1.0].index,
             portfolio.total[signal_data.positions == -1.0],
             'v', markersize=10, color='k', label='Sell')

    ax1.legend()
    plt.show()

if __name__ == "__main__":
    ticker = input("Enter the stock symbol (e.g., GOOG): ").upper()
    timeframe = input("Enter the timeframe for signals (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y): ")

    stock_data = fetch_data(ticker)
    signal_data = prepare_signals(stock_data)
    
    end_date = stock_data.index[-1]
    start_date = end_date - pd.DateOffset(days=int(timeframe[:-1]) * {'d': 1, 'mo': 30, 'y': 365}[timeframe[-1]])
    filtered_signal_data = signal_data.loc[start_date:end_date]

    plot_signals(filtered_signal_data, timeframe)
    
    portfolio = calculate_portfolio(signal_data)
    plot_portfolio(portfolio, signal_data)
