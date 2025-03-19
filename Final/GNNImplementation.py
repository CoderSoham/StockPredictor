import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam


# Fetch stock data
def fetch_stock_data(symbol, delta_minutes):
    df = yf.download(symbol, period="7d", interval=f"{delta_minutes}m")
    return df


# Preprocess data
def preprocess_stock_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']].values)

    X, y = [], []
    sequence_length = 60
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


# Build and train LSTM model
def train_lstm_model(X, y):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(units=50),
        Dense(units=1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    model.fit(X, y, epochs=10, batch_size=32)
    return model


# Prepare graph data for GNN
def prepare_graph_data(X):
    num_nodes = X.shape[0]
    edge_index = torch.tensor(
        [[i, i + 1] for i in range(num_nodes - 1)] +
        [[i + 1, i] for i in range(num_nodes - 1)], dtype=torch.long
    ).t()

    node_features = torch.tensor(X, dtype=torch.float).unsqueeze(-1)
    return Data(x=node_features, edge_index=edge_index)


# Define GNN model
class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels=1, out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Train GNN model
def train_gnn_model(stock_data):
    data = prepare_graph_data(stock_data)
    model = GNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out.squeeze(), data.x.squeeze())
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    return model


# Main function
def main():
    stock_symbol = input("Enter the stock symbol (e.g., GOOG): ")
    time_delta = int(input("Enter the prediction time delta in minutes (e.g., 5): "))

    print("Fetching stock data...")
    stock_data = fetch_stock_data(stock_symbol, time_delta)

    if stock_data.empty:
        print("No data found for the given symbol and time interval.")
        return

    print("Preprocessing stock data...")
    X, y, scaler = preprocess_stock_data(stock_data)

    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = y.reshape(-1, 1)

    print("Training LSTM model...")
    lstm_model = train_lstm_model(X, y)

    print("Preparing graph data and training GNN model...")
    gnn_model = train_gnn_model(X[:, :, 0])

    print("Models trained successfully!")


if __name__ == "__main__":
    main()
