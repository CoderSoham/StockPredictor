import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf 

def fetch_historical_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data['Close']

def calculate_parameters(historical_prices):
    log_returns = np.log(historical_prices / historical_prices.shift(1))
    mu = np.mean(log_returns)  # Drift
    sigma = np.std(log_returns)  # Volatility
    return mu, sigma

def simulate_stock_price_with_events(S0, mu, sigma, T, dt, event_prob, event_impact_range, N):
    paths = []
    for _ in range(N):
        t = np.linspace(0, T, int(T/dt))
        W = np.random.standard_normal(size=len(t))
        W = np.cumsum(W) * np.sqrt(dt)  # Brownian motion
        X = (mu - 0.5 * sigma**2) * t + sigma * W
        S = S0 * np.exp(X)

        # Introduce random events
        for i in range(1, len(S)):
            if np.random.rand() < event_prob:
                impact = np.random.uniform(event_impact_range[0], event_impact_range[1])
                S[i:] *= (1 + impact)
        paths.append(S)
    return paths

def plot_simulation_paths(paths):
    plt.figure(figsize=(10, 6))
    for path in paths:
        plt.plot(path)
    plt.title('Paths of the Stock Price Simulations')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.show()

# Fetch historical data
historical_prices = fetch_historical_data('AAPL', '2020-01-01', '2021-01-01')
mu, sigma = calculate_parameters(historical_prices)

# Parameters
S0 = historical_prices[-1]  
T = 1  
dt = 1/252
N = 100 
event_prob = 0.01
event_impact_range = (-0.2, 0.2)

# Run simulation
paths = simulate_stock_price_with_events(S0, mu, sigma, T, dt, event_prob, event_impact_range, N)
plot_simulation_paths(paths)
