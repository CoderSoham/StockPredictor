import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('apple.csv')

# Convert the 'Date' column to a datetime object
data['Date'] = pd.to_datetime(data['Date'])

# Prepare the features (X) and target (y) data
X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = data['Adj Close']

# Split the data into training and testing sets, including the 'Date' column
X_train, X_test, y_train, y_test = train_test_split(data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']], y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train[['Open', 'High', 'Low', 'Close', 'Volume']], y_train)

# Evaluate the model
train_rmse = mean_squared_error(y_train, model.predict(X_train[['Open', 'High', 'Low', 'Close', 'Volume']]), squared=False)
test_rmse = mean_squared_error(y_test, model.predict(X_test[['Open', 'High', 'Low', 'Close', 'Volume']]), squared=False)

print(f'Training RMSE: {train_rmse}')
print(f'Testing RMSE: {test_rmse}')

# Predict the prices for the test set
y_pred = model.predict(X_test[['Open', 'High', 'Low', 'Close', 'Volume']])

# Plot the actual prices
plt.plot(X_test['Date'], y_test, label='Actual Price', linestyle='-', color='blue')

# Plot the predicted prices
plt.plot(X_test['Date'], y_pred, label='Predicted Price', linestyle='--', color='red')

# Set the title and axis labels
plt.title('Apple Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')

# Set the figure size and font size
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams.update({'font.size': 12})

# Add a legend and show the plot
plt.legend()
plt.show()
