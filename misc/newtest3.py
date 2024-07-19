import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
from ta.trend import EMAIndicator
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

def fetch_data(symbol):
    # Fetch historical data for the stock
    stock_data = yf.download(symbol, period="1d", interval="1m")
    return stock_data

def train_model(stock_data):
    # Prepare features and target variables
    stock_data['EMA'] = EMAIndicator(stock_data['Close'], window=20).ema_indicator()
    stock_data = stock_data.dropna()

    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'EMA']]
    y_close = stock_data['Close']
    y_high = stock_data['High']
    y_low = stock_data['Low']

    # Train the Random Forest Regression models
    model_close = RandomForestRegressor(n_estimators=100, random_state=42)
    model_high = RandomForestRegressor(n_estimators=100, random_state=42)
    model_low = RandomForestRegressor(n_estimators=100, random_state=42)

    model_close.fit(X, y_close)
    model_high.fit(X, y_high)
    model_low.fit(X, y_low)

    return model_close, model_high, model_low

def predict_next_day_prices(model_close, model_high, model_low, last_data_point):
    # Predict the next day's closing price, high, and low
    next_day_open = last_data_point['Open'].iloc[-1]
    next_day_volume = last_data_point['Volume'].iloc[-1]
    next_day_ema = last_data_point['EMA'].iloc[-1]

    next_day_data = [[next_day_open, next_day_open, next_day_open, next_day_open, next_day_volume, next_day_ema]]

    next_day_close = model_close.predict(next_day_data)[0]
    next_day_high = model_high.predict(next_day_data)[0]
    next_day_low = model_low.predict(next_day_data)[0]

    return next_day_close, next_day_high, next_day_low

def generate_signals(stock_data):
    stock_data['Signal'] = np.where(stock_data['Close'] > stock_data['EMA'], 'Buy', 'Sell')
    return stock_data

app = dash.Dash(__name__)

app.layout = html.Div(
    style={'backgroundColor': '#1e1e1e', 'color': 'white', 'height': '100vh', 'margin': 0, 'display': 'flex'},
    children=[
        html.Div(
            style={'width': '20%', 'padding': '10px', 'backgroundColor': '#2e2e2e'},
            children=[
                html.H2('Stock Prediction Tool', style={'textAlign': 'center'}),
                dcc.Input(id='stock-symbol', type='text', placeholder='Enter stock symbol', value='GOOG', style={'marginBottom': '10px', 'width': '100%'}),
                html.Button('Submit', id='submit-button', n_clicks=0, style={'width': '100%', 'marginBottom': '10px'}),
                dcc.Input(id='ema-window', type='number', placeholder='EMA Window', value=20, style={'marginBottom': '10px', 'width': '100%'}),
                html.Button('Update', id='update-button', n_clicks=0, style={'width': '100%', 'marginBottom': '10px'}),
                html.Div(id='predicted-prices', style={'textAlign': 'center', 'padding': '10px', 'color': 'white'})
            ]
        ),
        html.Div(
            style={'width': '80%', 'padding': '10px'},
            children=[
                dcc.Graph(id='candlestick-chart'),
                dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0)
            ]
        )
    ]
)

@app.callback(
    [Output('candlestick-chart', 'figure'),
     Output('predicted-prices', 'children')],
    [Input('submit-button', 'n_clicks'),
     Input('update-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('stock-symbol', 'value'),
     State('ema-window', 'value')]
)
def update_chart(submit_n_clicks, update_n_clicks, n_intervals, symbol, ema_window):
    if not symbol:
        return {}, 'Please enter a stock symbol.'

    # Fetch and preprocess data
    stock_data = fetch_data(symbol)
    stock_data['EMA'] = EMAIndicator(stock_data['Close'], window=ema_window).ema_indicator()
    stock_data = generate_signals(stock_data)
    last_data_point = stock_data.tail(1)

    # Train model and predict prices
    model_close, model_high, model_low = train_model(stock_data)
    next_day_close, next_day_high, next_day_low = predict_next_day_prices(model_close, model_high, model_low, last_data_point)

    # Prepare candlestick data
    candlestick = go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='Candlestick'
    )

    # Prepare buy/sell signals
    signals = go.Scatter(
        x=stock_data.index,
        y=stock_data['Close'],
        mode='markers',
        marker=dict(
            color=np.where(stock_data['Signal'] == 'Buy', 'green', 'red'),
            size=8
        ),
        name='Buy/Sell Signals'
    )

    # Prepare EMA line
    ema_line = go.Scatter(
        x=stock_data.index,
        y=stock_data['EMA'],
        mode='lines',
        line=dict(color='orange', width=2),
        name='EMA'
    )

    layout = go.Layout(
        plot_bgcolor='rgb(30, 30, 30)',
        paper_bgcolor='rgb(30, 30, 30)',
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
    )

    fig = go.Figure(data=[candlestick, signals, ema_line], layout=layout)

    # Calculate market direction
    today_close = last_data_point['Close'].iloc[-1]
    if next_day_close > today_close:
        market_direction = "up"
    elif next_day_close < today_close:
        market_direction = "down"
    else:
        market_direction = "unchanged"
    percentage_change = ((next_day_close - today_close) / today_close) * 100

    predicted_prices_text = f"""
    Predicted Closing Price for {symbol} tomorrow: {next_day_close:.2f}
    Predicted High Price for {symbol} tomorrow: {next_day_high:.2f}
    Predicted Low Price for {symbol} tomorrow: {next_day_low:.2f}
    Market is expected to move {market_direction} by {percentage_change:.2f}% compared to today.
    """

    return fig, predicted_prices_text

if __name__ == '__main__':
    app.run_server(debug=True)
