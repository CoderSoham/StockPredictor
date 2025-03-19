import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Fetch historical data for price-related calculations and predictions
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data for better prediction
    hist = stock.history(start=start_date, end=end_date)
    
    return info, hist

def calculate_metrics(info, hist):
    metrics = {}
    
    # Basic info
    metrics['name'] = info.get('longName', 'N/A')
    metrics['current_price'] = info.get('currentPrice', None)
    
    # Price-related metrics
    metrics['fifty_day_average'] = info.get('fiftyDayAverage', None)
    metrics['two_hundred_day_average'] = info.get('twoHundredDayAverage', None)
    
    # Valuation metrics
    metrics['pe_ratio'] = info.get('trailingPE', None)
    metrics['forward_pe'] = info.get('forwardPE', None)
    metrics['peg_ratio'] = info.get('pegRatio', None)
    metrics['price_to_book'] = info.get('priceToBook', None)
    
    # Growth and profitability
    metrics['revenue_growth'] = info.get('revenueGrowth', None)
    metrics['profit_margins'] = info.get('profitMargins', None)
    
    # Dividend metrics
    metrics['dividend_yield'] = info.get('dividendYield', None)
    metrics['payout_ratio'] = info.get('payoutRatio', None)
    
    # Calculated metrics
    if not pd.isna(hist['Close'].iloc[-1]) and not pd.isna(hist['Close'].iloc[0]):
        metrics['yearly_return'] = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
    else:
        metrics['yearly_return'] = None
    
    return metrics

def predict_prices(hist, days=[1, 7, 30, 90]):
    df = hist.copy()
    df['Date'] = df.index
    df['Date'] = (df['Date'] - df['Date'].min()).dt.days
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
    X = df[features]
    y = df['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + day for day in days]
    future_features = X.iloc[-1:].copy()
    
    predictions = {}
    for day, future_date in zip(days, future_dates):
        future_features['Date'] = future_date
        prediction = model.predict(future_features)[0]
        predictions[day] = prediction
    
    return predictions

def analyze_stock(ticker):
    info, hist = get_stock_data(ticker)
    metrics = calculate_metrics(info, hist)
    predictions = predict_prices(hist)
    
    analysis = []
    score = 0
    
    # Analyze price trends
    if metrics['current_price'] and metrics['fifty_day_average'] and metrics['two_hundred_day_average']:
        if metrics['current_price'] < metrics['fifty_day_average'] < metrics['two_hundred_day_average']:
            analysis.append("The stock is trading below both its 50-day and 200-day moving averages, which could indicate it's undervalued.")
            score -= 1
        elif metrics['current_price'] > metrics['fifty_day_average'] > metrics['two_hundred_day_average']:
            analysis.append("The stock is trading above both its 50-day and 200-day moving averages, which could indicate it's overvalued.")
            score += 1
    
    # Analyze P/E ratio
    if metrics['pe_ratio']:
        if metrics['pe_ratio'] < 15:
            analysis.append(f"The P/E ratio ({metrics['pe_ratio']:.2f}) is relatively low, suggesting the stock might be undervalued.")
            score -= 1
        elif metrics['pe_ratio'] > 25:
            analysis.append(f"The P/E ratio ({metrics['pe_ratio']:.2f}) is relatively high, suggesting the stock might be overvalued.")
            score += 1
    
    # Analyze PEG ratio
    if metrics['peg_ratio']:
        if metrics['peg_ratio'] < 1:
            analysis.append(f"The PEG ratio ({metrics['peg_ratio']:.2f}) is below 1, indicating the stock might be undervalued relative to its growth rate.")
            score -= 1
        elif metrics['peg_ratio'] > 1.5:
            analysis.append(f"The PEG ratio ({metrics['peg_ratio']:.2f}) is above 1.5, indicating the stock might be overvalued relative to its growth rate.")
            score += 1
    
    # Analyze Price-to-Book ratio
    if metrics['price_to_book']:
        if metrics['price_to_book'] < 1:
            analysis.append(f"The Price-to-Book ratio ({metrics['price_to_book']:.2f}) is below 1, suggesting the stock might be undervalued.")
            score -= 1
        elif metrics['price_to_book'] > 3:
            analysis.append(f"The Price-to-Book ratio ({metrics['price_to_book']:.2f}) is relatively high, suggesting the stock might be overvalued.")
            score += 1
    
    # Analyze dividend yield
    if metrics['dividend_yield']:
        if metrics['dividend_yield'] > 0.04:  # 4% dividend yield
            analysis.append(f"The dividend yield ({metrics['dividend_yield']:.2%}) is relatively high, which could indicate undervaluation.")
            score -= 1
    
    # Final assessment
    if score < 0:
        verdict = "UNDERVALUED"
    elif score > 0:
        verdict = "OVERVALUED"
    else:
        verdict = "FAIRLY VALUED"
    
    return verdict, analysis, metrics, predictions

def main():
    ticker = input("Enter the stock ticker symbol: ").upper()
    
    try:
        verdict, analysis, metrics, predictions = analyze_stock(ticker)
        
        print(f"\nAnalysis for {ticker} - {metrics['name']}:")
        print(f"Current Price: ${metrics['current_price']:.2f}")
        print(f"Verdict: The stock appears to be {verdict}")
        
        print("\nPrice Predictions:")
        for day, price in predictions.items():
            print(f"  {day} day{'s' if day > 1 else ''}: ${price:.2f}")
        
        print("\nReasoning:")
        for point in analysis:
            print(f"- {point}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check the ticker symbol and try again.")

if __name__ == "__main__":
    main()