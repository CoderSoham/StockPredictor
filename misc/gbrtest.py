"""
In this code:

We're using a Gradient Boosting Regressor instead of a Random Forest Regressor.
We're performing hyperparameter tuning using GridSearchCV to find the best parameters for the model.
We calculate the error percentage between the predicted price and the real-time price.
We include the error percentage in the plot title.
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import time
import sys

def fetch_data_and_train_model(symbol):
    # Fetch historical data for the stock
    stock_data = yf.download(symbol, period="max")
    
    # Convert 'Date' column to datetime
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    
    # Split data into features and target variable
    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = stock_data['Adj Close']
    
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }
    
    # Initialize the Gradient Boosting Regressor
    model = GradientBoostingRegressor(random_state=42)
    
    # Perform GridSearchCV to find the best parameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    
    # Get the best model from GridSearchCV
    best_model = grid_search.best_estimator_
    
    return best_model

def predict_and_plot_realtime_price(model, symbol):
    """
    Predict and plot real-time stock price using the trained model.
    
    Args:
    - model: Trained regression model
    - symbol: Stock symbol
    """
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.xlabel('Time')
    plt.ylabel('Price')
    
    # Initialize lists to store data points
    timestamps = []
    real_prices = []
    predicted_prices = []
    
    # Initialize variables to store previous data
    prev_timestamp = None
    prev_real_price = None
    prev_predicted_price = None
    
    # Infinite loop to continuously update the plot with real-time data
    try:
        while True:
            # Fetch real-time data
            stock = yf.Ticker(symbol)
            realtime_data = stock.history(period='1d')
    
            # Extract the latest timestamp and price
            latest_timestamp = datetime.now()
            latest_price = realtime_data['Close'].iloc[-1]
    
            # Predict price using the model
            predicted_price = model.predict([[
                realtime_data['Open'].iloc[-1], 
                realtime_data['High'].iloc[-1], 
                realtime_data['Low'].iloc[-1], 
                realtime_data['Close'].iloc[-1], 
                realtime_data['Volume'].iloc[-1]
            ]])[0]
            
            # Append data to lists
            timestamps.append(latest_timestamp)
            real_prices.append(latest_price)
            predicted_prices.append(predicted_price)
            
            # Calculate error percentage
            error_percentage = abs((predicted_price - latest_price) / latest_price) * 100
            
            # Plot the real-time and predicted prices
            plt.plot(timestamps, real_prices, label='Real-Time Price', color='green', marker='o')
            plt.plot(timestamps, predicted_prices, label='Predicted Price', linestyle='--', color='red', marker='x')
            
            plt.title(f'Real-Time and Predicted Stock Price for {symbol}\nError Percentage: {error_percentage:.2f}%')
            plt.legend().remove()  # Remove legend
            plt.pause(0.01)
                
            # Update previous data
            prev_timestamp = latest_timestamp
            prev_real_price = latest_price
            prev_predicted_price = predicted_price
    
            # Wait for 10 seconds before fetching the next data
            time.sleep(10)
    except KeyboardInterrupt:
        print("Exiting program...")
        sys.exit(0)
    except Exception as e:
        print("Error:", e)

def main():
    # Ask user for stock symbol
    symbol = input("Enter the stock symbol (e.g., GOOG): ").upper()
    
    # Fetch data and train model
    model = fetch_data_and_train_model(symbol)
    
    # Predict and plot real-time price
    predict_and_plot_realtime_price(model, symbol)

if __name__ == "__main__":
    main()
