import streamlit as st
import yfinance as yf
from datetime import datetime

def fetch_stock_data(symbol, start_date, end_date):
    try:
        # Fetch historical data for the selected stock within the specified time range
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data for symbol {symbol}: {e}")
        return None

def calculate_returns(stock_data, investment_amount, start_date, end_date):
    try:
        # Check if the selected dates are within the range of the stock data
        if start_date not in stock_data.index or end_date not in stock_data.index:
            st.error("Invalid date range selected. Please select dates within the available stock data.")
            return None

        # Get the stock price on the start date and end date
        start_price = stock_data.loc[str(start_date)]['Close']
        end_price = stock_data.loc[str(end_date)]['Close']

        # Calculate the number of shares bought
        shares_bought = investment_amount / start_price

        # Calculate the returns
        returns = shares_bought * end_price - investment_amount

        return returns
    except KeyError as e:
        st.error("Invalid date range selected. Please select valid dates.")
        return None

def main():
    # Set page title
    st.title("Investment Returns Calculator")

    # Sidebar - User input
    st.sidebar.title("Input Parameters")
    symbol = st.sidebar.text_input("Enter the stock symbol (e.g., AAPL):").upper()
    investment_amount = st.sidebar.number_input("Enter the investment amount ($):", value=1000, step=100)
    start_date = st.sidebar.text_input("Enter the start date (YYYY-MM-DD):")
    end_date = st.sidebar.text_input("Enter the end date (YYYY-MM-DD):")

    # Convert dates to datetime objects
    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        st.error("Invalid date format. Please use the format YYYY-MM-DD.")
        return

    # Fetch stock data
    stock_data = fetch_stock_data(symbol, start_date, end_date)

    # Calculate and display returns
    if stock_data is not None:
        st.subheader("Stock Data")
        st.write(stock_data)

        if st.sidebar.button("Calculate Returns"):
            returns = calculate_returns(stock_data, investment_amount, start_date, end_date)
            if returns is not None:
                st.success(f"Your returns after investing ${investment_amount} in {symbol} from {start_date} to {end_date} are ${returns:.2f}")

if __name__ == "__main__":
    main()
