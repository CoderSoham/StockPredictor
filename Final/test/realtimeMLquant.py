import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Fetch real-time prices for the last 7 days with 5-min intervals
def fetch_data(ticker):
    stock_data = yf.download(ticker, period='7d', interval='5m')
    stock_data['Returns'] = stock_data['Close'].pct_change()
    stock_data = stock_data.dropna()
    return stock_data

# 2. Feature Engineering for ML models using all OHLCV values
def feature_engineering(data):
    data['SMA10'] = data['Close'].rolling(window=10).mean()
    data['SMA30'] = data['Close'].rolling(window=30).mean()
    data['RSI'] = compute_rsi(data['Close'])

    # Include OHLCV data and lagged values for better prediction
    data['Open_Lag1'] = data['Open'].shift(1)
    data['High_Lag1'] = data['High'].shift(1)
    data['Low_Lag1'] = data['Low'].shift(1)
    data['Close_Lag1'] = data['Close'].shift(1)
    data['Volume_Lag1'] = data['Volume'].shift(1)
    
    data = data.dropna()
    return data

# 3. RSI Calculation
def compute_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 4. Train ML models and predict buy/sell signals
def train_models(data):
    print("Starting feature extraction and model training...\n")
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA10', 'SMA30', 'RSI',
                'Open_Lag1', 'High_Lag1', 'Low_Lag1', 'Close_Lag1', 'Volume_Lag1']
    
    X = data[features].values
    y = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)  # 1 for Buy, 0 for Sell
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"Training Random Forest Classifier on {len(X_train)} data points...\n")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    data['Prediction_RF'] = rf.predict(X)
    
    print(f"Training Support Vector Classifier on {len(X_train)} data points...\n")
    svc = SVC()
    svc.fit(X_train, y_train)
    data['Prediction_SVC'] = svc.predict(X)
    
    return data

# 5. Simulate trading with buy/sell signals and show detailed calculations
def simulate_trading(data, initial_balance=10000):
    print("Starting trading simulation...\n")
    balance = initial_balance
    position = 0  # No stock initially
    for index, row in data.iterrows():
        print(f"Time: {index}")
        print(f"Close Price: {row['Close']:.2f}")
        
        if row['Prediction_RF'] == 1:  # Buy signal
            if position == 0:  # If no current position
                position = balance / row['Close']  # Buy full position
                balance = 0
                print(f"Buying {position:.2f} units at price {row['Close']:.2f}")
        
        elif row['Prediction_RF'] == 0:  # Sell signal
            if position > 0:  # If holding stock
                balance = position * row['Close']  # Sell everything
                print(f"Selling {position:.2f} units at price {row['Close']:.2f}")
                position = 0
        
        print(f"Balance: ${balance:.2f}\n")
    
    # At the end, sell any remaining positions
    if position > 0:
        balance = position * data.iloc[-1]['Close']
    
    print(f"Final Balance: ${balance:.2f}")
    profit_loss = balance - initial_balance
    print(f"Profit/Loss after 7 days: ${profit_loss:.2f}")
    return profit_loss

# 6. Plot candlestick chart with buy/sell signals and smoother curve
def plot_trading_signals(data):
    data['SMA_Smooth'] = data['Close'].rolling(window=10).mean()
    
    # Candlestick plot
    mc = mpf.make_marketcolors(up='g', down='r', volume='in', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)
    add_plot = [mpf.make_addplot(data['SMA_Smooth'], color='blue')]
    
    buy_signals = data[data['Prediction_RF'] == 1].index
    sell_signals = data[data['Prediction_RF'] == 0].index
    
    signal_markers = {
        'buy': mpf.make_addplot(data['Close'], scatter=True, markersize=100, marker='^', color='green', alpha=0.8),
        'sell': mpf.make_addplot(data['Close'], scatter=True, markersize=100, marker='v', color='red', alpha=0.8),
    }

    mpf.plot(data, type='candle', style=s, volume=True, addplot=add_plot + [signal_markers['buy'], signal_markers['sell']], 
             title="Stock Price with Buy/Sell Signals", ylabel='Price', ylabel_lower='Volume')

# Run the full simulation and print output at each step
def run_simulation(ticker):
    data = fetch_data(ticker)
    data = feature_engineering(data)
    data = train_models(data)
    
    profit_loss = simulate_trading(data)
    plot_trading_signals(data)

# Example Usage
run_simulation('AAPL')
