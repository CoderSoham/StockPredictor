import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime


# Fetch historical stock data
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Prepare data for Transformer
def prepare_data(data, input_window, output_window, scaler):
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(data_scaled) - input_window - output_window):
        X.append(data_scaled[i:i+input_window])
        y.append(data_scaled[i+input_window:i+input_window+output_window])
    return np.array(X), np.array(y)

# Transformer Model
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerTimeSeries, self).__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(model_dim, input_dim)

    def forward(self, src):
        src = self.input_fc(src) + self.positional_encoding[:, :src.size(1), :]
        output = self.transformer_encoder(src)
        output = self.output_fc(output)
        return output

# Training function
def train_model(model, criterion, optimizer, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Prediction function
def predict(model, past_data, input_window, scaler):
    model.eval()
    with torch.no_grad():
        past_data = torch.tensor(past_data, dtype=torch.float32).unsqueeze(0)
        prediction = model(past_data)
        prediction = prediction.squeeze().numpy()
        prediction_reshaped = prediction.reshape(-1, 1)
        prediction_inversed = scaler.inverse_transform(prediction_reshaped).flatten()
    return prediction_inversed

# Visualization function
def plot_predictions(actual, predicted, title):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Prices')
    plt.plot(predicted, label='Predicted Prices')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def main():
    # Parameters
    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    input_window = 30
    output_window = 1
    model_dim = 64
    num_heads = 4
    num_layers = 2
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001

    # Fetch and prepare data
    stock_data = fetch_stock_data(symbol, start_date, end_date)
    close_prices = stock_data[['Close']].values
    scaler = MinMaxScaler()
    X, y = prepare_data(close_prices, input_window, output_window, scaler)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, criterion, and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, criterion, optimizer, train_loader, num_epochs)

    # Make predictions
    predictions = []
    for i in range(input_window, len(close_prices) - input_window):
        past_data = close_prices[i - input_window:i]
        predicted = predict(model, past_data, input_window, scaler)
        predictions.append(predicted)

    # Plot results
    plot_predictions(close_prices[input_window:], predictions, f'{symbol} Stock Price Prediction')

if __name__ == "__main__":
    main()
