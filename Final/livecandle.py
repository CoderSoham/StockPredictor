import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

# Initialize Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(style={'backgroundColor': '#2E2E2E', 'color': '#FFFFFF'}, children=[
    html.H1(children='Stock Dashboard', style={'textAlign': 'center'}),
    
    html.Div([
        html.Label('Enter Stock Symbol:'),
        dcc.Input(id='stock-symbol', value='GOOG', type='text', style={'marginRight': '10px'}),
        
        html.Label('Select Technical Indicators:'),
        dcc.Checklist(
            id='indicators',
            options=[
                {'label': 'SMA 20', 'value': 'sma20'},
                {'label': 'SMA 50', 'value': 'sma50'},
                {'label': 'EMA 20', 'value': 'ema20'},
                {'label': 'EMA 50', 'value': 'ema50'}
            ],
            value=['sma20'],  # Default selection
            style={'marginBottom': '10px'}
        ),
        
        html.Button('Submit', id='submit-button', n_clicks=0, style={'backgroundColor': '#007bff', 'color': '#FFFFFF'}),
    ], style={'padding': '10px'}),
    
    dcc.Graph(id='live-candlestick', style={'height': '70vh'}),
    dcc.Graph(id='indicator-graph', style={'height': '70vh'}),
])

def fetch_live_data(symbol):
    stock = yf.Ticker(symbol)
    minute_data = stock.history(period="1d", interval="1m")
    return minute_data

def calculate_indicators(data, indicators):
    if 'sma20' in indicators:
        data['SMA20'] = data['Close'].rolling(window=20).mean()
    if 'sma50' in indicators:
        data['SMA50'] = data['Close'].rolling(window=50).mean()
    if 'ema20' in indicators:
        data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
    if 'ema50' in indicators:
        data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
    return data

@app.callback(
    [Output('live-candlestick', 'figure'),
     Output('indicator-graph', 'figure')],
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('stock-symbol', 'value'),
     dash.dependencies.State('indicators', 'value')]
)
def update_graph(n_clicks, symbol, indicators):
    if not symbol:
        symbol = 'GOOG'
    
    data = fetch_live_data(symbol)
    data.index = pd.to_datetime(data.index)
    
    # Calculate indicators
    data = calculate_indicators(data, indicators)
    
    # Create the candlestick chart
    fig1 = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=symbol,
        line=dict(color='rgba(255, 255, 255, 0.8)'),
        increasing_line_color='rgba(76, 175, 80, 0.8)', # Green
        decreasing_line_color='rgba(244, 67, 54, 0.8)'  # Red
    )])
    
    fig1.update_layout(
        title=f'{symbol} Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        paper_bgcolor='#2E2E2E',
        plot_bgcolor='#1E1E1E',
        font=dict(color='#FFFFFF')
    )
    
    # Create the indicator chart
    fig2 = go.Figure()
    fig2.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=symbol,
        line=dict(color='rgba(255, 255, 255, 0.8)'),
        increasing_line_color='rgba(76, 175, 80, 0.8)',
        decreasing_line_color='rgba(244, 67, 54, 0.8)'
    ))
    
    if 'sma20' in indicators:
        fig2.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='rgba(255, 215, 0, 0.8)')  # Yellow
        ))
    if 'sma50' in indicators:
        fig2.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='rgba(0, 255, 255, 0.8)')  # Cyan
        ))
    if 'ema20' in indicators:
        fig2.add_trace(go.Scatter(
            x=data.index,
            y=data['EMA20'],
            mode='lines',
            name='EMA 20',
            line=dict(color='rgba(255, 0, 255, 0.8)')  # Magenta
        ))
    if 'ema50' in indicators:
        fig2.add_trace(go.Scatter(
            x=data.index,
            y=data['EMA50'],
            mode='lines',
            name='EMA 50',
            line=dict(color='rgba(0, 255, 0, 0.8)')  # Lime
        ))
    
    fig2.update_layout(
        title=f'{symbol} with Indicators',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        paper_bgcolor='#2E2E2E',
        plot_bgcolor='#1E1E1E',
        font=dict(color='#FFFFFF')
    )
    
    return fig1, fig2

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
