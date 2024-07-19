import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import yfinance as yf

def fetch_data(symbol):
    # Fetch historical data for the stock
    stock_data = yf.download(symbol, period="max")
    return stock_data

def train_model(stock_data):
    # Split the data into features (X) and target variable (y)
    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = stock_data['Adj Close']

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest Regression model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

def test_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE) to evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    accuracy = (1 - mse / y_test.var()) * 100
    return accuracy

def main():
    # Fetch historical data for Google
    symbol = 'GOOG'
    stock_data = fetch_data(symbol)

    # Train the model
    model, X_test, y_test = train_model(stock_data)

    # Test the model
    accuracy = test_model(model, X_test, y_test)

    print(f"Accuracy of the Random Forest Regression model on predicting Google's stock prices: {accuracy:.2f}%\n")

    # Make predictions for the last 10 data points
    last_10_predictions = model.predict(X_test[-10:])
    real_prices = y_test[-10:]

    print("Last 10 Predictions vs Real Prices:")
    print("Predicted Price\t\tReal Price")
    for i in range(10):
        print(f"{last_10_predictions[i]:.2f}\t\t\t{real_prices.iloc[i]:.2f}")

if __name__ == "__main__":
    main()
