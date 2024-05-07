import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import streamlit as st
import numpy as np

def fetch_data(symbol):
    try:
        # Fetch historical data for the stock
        stock_data = yf.download(symbol, period="max")
        
        # Convert 'Date' column to datetime
        stock_data.reset_index(inplace=True)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data for symbol {symbol}: {e}")
        return None

def predict_price(stock_data, n_seconds):
    try:
        # Get the latest timestamp and price
        latest_timestamp = datetime.now()
        latest_price = stock_data['Close'].iloc[-1]
        
        # Predict price n seconds in the future
        predicted_timestamp = latest_timestamp + timedelta(seconds=n_seconds)
        predicted_price = np.random.uniform(latest_price * 0.95, latest_price * 1.05) # Dummy prediction
        
        return predicted_timestamp, predicted_price
    except Exception as e:
        st.error(f"Error predicting price: {e}")
        return None, None

def main():
    # Sidebar dropdown to choose mode
    mode = st.sidebar.selectbox("Choose mode", ["Live and Predicted Side by Side", "Predict 10 Seconds in Future"])
    
    # Ask user for stock symbol
    symbol = st.sidebar.text_input("Enter the stock symbol (e.g., GOOG): ").upper()
    
    # Fetch data
    stock_data = fetch_data(symbol)
    
    if stock_data is not None:
        if mode == "Live and Predicted Side by Side":
            # Live and predicted side by side mode
            # Initialize variables
            timestamps = []
            real_prices = []
            predicted_prices = []
            error_percentages = []
            
            # Create a streamlit line chart
            chart = st.line_chart()
            
            # Infinite loop to continuously update the plot with real-time data
            try:
                while True:
                    # Fetch real-time data
                    stock = yf.Ticker(symbol)
                    realtime_data = stock.history(period='1d')
            
                    # Extract the latest timestamp and price
                    latest_timestamp = datetime.now()
                    latest_price = realtime_data['Close'].iloc[-1]
            
                    # Predict price using the Random Forest Regression model
                    predicted_price = np.random.uniform(latest_price * 0.95, latest_price * 1.05) # Dummy prediction
                    
                    # Calculate error percentage
                    error_percentage = ((latest_price - predicted_price) / latest_price) * 100
                    
                    # Append data to lists
                    timestamps.append(latest_timestamp)
                    real_prices.append(latest_price)
                    predicted_prices.append(predicted_price)
                    error_percentages.append(error_percentage)
                    
                    # Update chart data
                    chart_data = pd.DataFrame({
                        'Timestamp': timestamps,
                        'Real Price': real_prices,
                        'Predicted Price': predicted_prices
                    })
                    
                    # Display the chart
                    chart.line_chart(chart_data.set_index('Timestamp'))
                    
                    # Display error percentage
                    st.write(f"Error Percentage: {error_percentage:.2f}%")
                    
                    # Wait for 10 seconds before fetching the next data
                    time.sleep(10)
            except KeyboardInterrupt:
                print("Exiting program...")
        
        elif mode == "Predict 10 Seconds in Future":
            # Predict 10 seconds in the future mode
            n_seconds = 10
            
            # Create a streamlit line chart
            chart = st.line_chart()
            
            # Infinite loop to continuously update the plot with real-time data
            try:
                while True:
                    # Fetch real-time data
                    stock = yf.Ticker(symbol)
                    realtime_data = stock.history(period='1d')
            
                    # Predict price 10 seconds in the future
                    predicted_timestamp, predicted_price = predict_price(stock_data, n_seconds)
                    
                    if predicted_timestamp is not None and predicted_price is not None:
                        # Extract the latest timestamp and price
                        latest_timestamp = datetime.now()
                        latest_price = realtime_data['Close'].iloc[-1]
                        
                        # Calculate error percentage
                        error_percentage = ((latest_price - predicted_price) / latest_price) * 100
                        
                        # Update chart data
                        chart_data = pd.DataFrame({
                            'Timestamp': [latest_timestamp, predicted_timestamp],
                            'Price': [latest_price, predicted_price]
                        })
                        
                        # Display the chart
                        chart.line_chart(chart_data.set_index('Timestamp'))
                        
                        # Display error percentage
                        st.write(f"Error Percentage: {error_percentage:.2f}%")
                    
                    # Wait for 10 seconds before fetching the next data
                    time.sleep(10)
            except KeyboardInterrupt:
                print("Exiting program...")

if __name__ == "__main__":
    main()
