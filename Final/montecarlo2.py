import numpy as np
import matplotlib.pyplot as plt

def simulate_stock_price_with_events(S0, mu, sigma, T, dt, event_prob, event_impact_range):
    N = int(T / dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)  # Brownian motion
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)

    # Introduce random events
    for i in range(1, N):
        if np.random.rand() < event_prob:
            impact = np.random.uniform(event_impact_range[0], event_impact_range[1])
            S[i:] *= (1 + impact)  # Apply the event's impact to the stock price

    return S

def plot_simulation_paths(S0, mu, sigma, T, dt, N, event_prob, event_impact_range, num_paths):
    plt.figure(figsize=(10, 6))
    for _ in range(num_paths):
        prices = simulate_stock_price_with_events(S0, mu, sigma, T, dt, event_prob, event_impact_range)
        plt.plot(prices)

    plt.title('Paths of the Stock Price Simulations')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.show()

# Parameters
S0 = 100  # Initial stock price
mu = 0.05  # Expected return
sigma = 0.2  # Volatility
T = 1  # Total time in years
dt = 1/252  # Time step (daily steps assuming 252 trading days per year)
N = 10000  # Number of simulations
event_prob = 0.01  # Probability of an event at each step
event_impact_range = (-0.2, 0.2)  # Range of event impacts
num_paths = 100  # Number of paths to plot

plot_simulation_paths(S0, mu, sigma, T, dt, N, event_prob, event_impact_range, num_paths)
