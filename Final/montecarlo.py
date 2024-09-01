import numpy as np
import matplotlib.pyplot as plt

def simulate_stock_price_with_events(S0, mu, sigma, T, dt, event_prob, event_impact_range):
    """
    Simulates the stock price using Geometric Brownian Motion with random events.

    Parameters:
    S0 (float): Initial stock price.
    mu (float): Expected return.
    sigma (float): Volatility.
    T (int): Total time in days.
    dt (float): Time increment (e.g., 1/252 for daily steps if 252 trading days in a year).
    event_prob (float): Probability of a random event occurring at each time step.
    event_impact_range (tuple): Range of impact as a percentage (e.g., (-0.2, 0.2) for -20% to +20%).

    Returns:
    np.ndarray: Simulated stock prices with random events.
    """
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

def monte_carlo_simulation(S0, mu, sigma, T, dt, N, event_prob, event_impact_range):
    """
    Monte Carlo simulation for stock prices with random events.

    Parameters:
    S0 (float): Initial stock price.
    mu (float): Expected return.
    sigma (float): Volatility.
    T (int): Total time in years.
    dt (float): Time increment (e.g., 1/252 for daily steps if 252 trading days in a year).
    N (int): Number of simulations.
    event_prob (float): Probability of a random event occurring at each time step.
    event_impact_range (tuple): Range of impact as a percentage (e.g., (-0.2, 0.2) for -20% to +20%).

    Returns:
    np.ndarray: Array of final stock prices from all simulations.
    """
    final_prices = np.zeros(N)

    for i in range(N):
        prices = simulate_stock_price_with_events(S0, mu, sigma, T, dt, event_prob, event_impact_range)
        final_prices[i] = prices[-1]

    return final_prices

# Example usage:
S0 = 100  # Initial stock price
mu = 0.05  # Expected return
sigma = 0.2  # Volatility
T = 1  # Total time in years
dt = 1/252  # Time step (daily steps assuming 252 trading days per year)
N = 10000  # Number of simulations
event_prob = 0.01  # 1% chance of a random event occurring at each time step
event_impact_range = (-0.2, 0.2)  # Event can cause -20% to +20% change in price

final_prices = monte_carlo_simulation(S0, mu, sigma, T, dt, N, event_prob, event_impact_range)

# Plotting the distribution of final prices
plt.hist(final_prices, bins=50, alpha=0.75)
plt.title('Distribution of Final Stock Prices (Monte Carlo Simulation)')
plt.xlabel('Final Stock Price')
plt.ylabel('Frequency')
plt.show()
