from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
from datetime import datetime
import json
import time

app = Flask(__name__)

# Load historical data for Google
google_data = pd.read_csv('google.csv')

# Convert 'Date' column to datetime
google_data['Date'] = pd.to_datetime(google_data['Date'])

# Initialize RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker_symbol = request.form['ticker']
    
    if not ticker_symbol:
        return jsonify({'error': 'Ticker symbol cannot be empty'})
    
    try:
        google_stock = yf.Ticker(ticker_symbol)
        google_realtime_data = google_stock.history(period='1d')
        
        if google_realtime_data.empty:
            return jsonify({'error': 'No price data found for the ticker symbol'})
        
        # Train the model with historical data
        model.fit(google_data[['Open', 'High', 'Low', 'Close', 'Volume']], google_data['Adj Close'])

        # Predict price
        prediction = model.predict([[google_realtime_data['Open'].iloc[-1], google_realtime_data['High'].iloc[-1], 
                                     google_realtime_data['Low'].iloc[-1], google_realtime_data['Close'].iloc[-1], 
                                     google_realtime_data['Volume'].iloc[-1]]])[0]

        # Prepare data for plotting
        data = {
            'realtime_price': google_realtime_data['Close'].tolist(),
            'predicted_price': [None] * (len(google_realtime_data) - 1) + [prediction]
        }
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
