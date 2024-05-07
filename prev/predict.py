import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = pd.read_csv('reliance.csv')


data['Date'] = pd.to_datetime(data['Date'])


X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = data['Adj Close']


X_train, X_test, y_train, y_test = train_test_split(data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']], y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train[['Open', 'High', 'Low', 'Close', 'Volume']], y_train)


train_rmse = mean_squared_error(y_train, model.predict(X_train[['Open', 'High', 'Low', 'Close', 'Volume']]), squared=False)
test_rmse = mean_squared_error(y_test, model.predict(X_test[['Open', 'High', 'Low', 'Close', 'Volume']]), squared=False)

print(f'Training RMSE: {train_rmse}')
print(f'Testing RMSE: {test_rmse}')


y_pred = model.predict(X_test[['Open', 'High', 'Low', 'Close', 'Volume']])


plt.plot(X_test['Date'], y_test, label='Actual Price', linestyle='-', color='blue')


plt.plot(X_test['Date'], y_pred, label='Predicted Price', linestyle='--', color='red')


plt.title('Apple Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')


plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams.update({'font.size': 12})


plt.legend()
plt.show()
