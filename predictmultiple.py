import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read data from CSV, skipping the first two rows to handle the multi-level header
data = pd.read_csv('4comp24yr.csv', skiprows=[0, 1])

# Use the first row as column headers
data.columns = data.iloc[0]

# Remove the first row, as it is now redundant
data = data.drop(0)

# Reset index after dropping the first row
data = data.reset_index(drop=True)

# Convert 'Date' column to datetime
data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])

# Specify the 'Ticker' column by its position (assuming it's the second column)
ticker_column = data.columns[1]

# Check if 'Ticker' column is present
if ticker_column not in data.columns:
    raise ValueError("No 'Ticker' column found in the DataFrame. Please check the CSV file.")

# Define tickers
tickers = data[ticker_column].unique()

# Initialize dictionaries to store models and RMSE values for each company
models = {}
train_rmse = {}
test_rmse = {}

# Loop through each ticker
for ticker in tickers:
    # Filter data for the current ticker
    ticker_data = data[data[ticker_column] == ticker]
    
    # Define features (X) and target variable (y)
    X = ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = ticker_data['Adj Close']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate RMSE for training and testing sets
    train_rmse[ticker] = mean_squared_error(y_train, model.predict(X_train), squared=False)
    test_rmse[ticker] = mean_squared_error(y_test, model.predict(X_test), squared=False)
    
    # Store the trained model
    models[ticker] = model

# Print RMSE values for each company
for ticker in tickers:
    print(f'Ticker: {ticker}')
    print(f'Training RMSE: {train_rmse[ticker]}')
    print(f'Testing RMSE: {test_rmse[ticker]}')
    print()

# Plot actual and predicted prices for one of the companies (e.g., first ticker)
ticker_to_plot = tickers[0]
model_to_plot = models[ticker_to_plot]

# Filter data for the chosen company
ticker_data_to_plot = data[data[ticker_column] == ticker_to_plot]

# Predictions
y_pred = model_to_plot.predict(ticker_data_to_plot[['Open', 'High', 'Low', 'Close', 'Volume']])

# Plot
plt.plot(ticker_data_to_plot.iloc[:, 0], ticker_data_to_plot['Adj Close'], label='Actual Price', linestyle='-', color='blue')
plt.plot(ticker_data_to_plot.iloc[:, 0], y_pred, label='Predicted Price', linestyle='--', color='red')
plt.title(f'{ticker_to_plot} Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams.update({'font.size': 12})
plt.legend()
plt.show()
