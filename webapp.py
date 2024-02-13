import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import streamlit as st

def main():
    st.title('Stock Price Visualizer')

    # User input for ticker
    ticker_input = st.text_input("Enter the ticker:", 'AAPL')

    # Date range
    start = dt.datetime(2010, 9, 30)
    end = dt.datetime(2024, 2, 5)

    # Fetching data
    data = yf.download(ticker_input, start=start, end=end)

    # Plotting
    st.pyplot(plot_stock_data(data, ticker_input))


def plot_stock_data(data, ticker):
    plt.figure(figsize=(12, 8))
    plt.plot(data.index, data['Close'], label=ticker)
    plt.title(f'Closing Prices of {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(False)
    return plt


if __name__ == "__main__":
    main()
