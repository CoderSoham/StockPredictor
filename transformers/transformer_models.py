import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Dense, LSTM, Conv1D, Flatten, Dropout,
                                     Reshape, Concatenate, AveragePooling1D, Lambda,
                                     LayerNormalization, Add, TimeDistributed,
                                     MultiHeadAttention)
import tensorflow.keras.backend as K

symbols = '^GSPC' 
data = yf.download(symbols, start='2010-01-01', end='2023-12-31')['Close']
data = data.dropna()
series = data['^GSPC'].values.reshape(-1, 1)
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)

def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 30
X, y = create_sequences(series_scaled, seq_length=SEQ_LENGTH)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mda = np.mean(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])) * 100
    return mae, rmse, mda

def arima_forecast(train, test):
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    return forecast

def build_lstm(seq_length, n_features):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, n_features)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_cnn(seq_length, n_features):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, n_features)),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_hybrid_cnn_lstm(seq_length, n_features):
    model = Sequential([ 
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(seq_length, n_features)),
        Dropout(0.1),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        Dropout(0.1),
        
        LSTM(50, activation='tanh', return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

class SeriesDecomposition(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, **kwargs):
        super(SeriesDecomposition, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.pool = AveragePooling1D(pool_size=kernel_size, strides=1, padding='same')
    def call(self, inputs):
        trend = self.pool(inputs)
        seasonal = inputs - trend
        return trend, seasonal
def build_sdtp(seq_length, n_features):
    inputs = Input(shape=(seq_length, n_features))
    trend, seasonal = SeriesDecomposition(kernel_size=3)(inputs)
    attn_out = MultiHeadAttention(num_heads=4, key_dim=32)(seasonal, seasonal)
    x = Concatenate()([trend, attn_out])
    x = Flatten()(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_numhtml(seq_length, n_features):
    segment_length = 10
    num_segments = seq_length // segment_length  
    inputs = Input(shape=(seq_length, n_features))
    x = Reshape((num_segments, segment_length, n_features))(inputs)

    def transformer_block(x_segment):
        self_attention = MultiHeadAttention(num_heads=2, key_dim=16)
        attn = self_attention(x_segment, x_segment)
        x_res = Add()([x_segment, attn])
        x_norm = LayerNormalization()(x_res)
        ffn = Dense(32, activation='relu')(x_norm)
        ffn = Dense(segment_length, activation='linear')(ffn)
        x_res2 = Add()([x_norm, ffn])
        return LayerNormalization()(x_res2)

    x = TimeDistributed(tf.keras.layers.Lambda(transformer_block))(x)
    x = Flatten()(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == '__main__':
    X_train_tf = X_train.astype(np.float32)
    y_train_tf = y_train.astype(np.float32)
    X_test_tf  = X_test.astype(np.float32)
    y_test_tf  = y_test.astype(np.float32)

    results = {}

    train_series = series_scaled[:len(series_scaled) - len(X_test)]
    test_series  = series_scaled[len(series_scaled) - len(X_test):]
    arima_pred = arima_forecast(train_series.flatten(), test_series.flatten())
    mae, rmse, mda = evaluate_metrics(test_series.flatten(), arima_pred)
    results['ARIMA'] = (mae, rmse, mda)
    print('ARIMA done.')
    
    lstm_model = build_lstm(SEQ_LENGTH, 1)
    lstm_model.fit(X_train_tf, y_train_tf, epochs=20, batch_size=16, verbose=0)
    lstm_pred = lstm_model.predict(X_test_tf).flatten()
    mae, rmse, mda = evaluate_metrics(y_test_tf.flatten(), lstm_pred)
    results['LSTM'] = (mae, rmse, mda)
    print('LSTM done.')

    cnn_model = build_cnn(SEQ_LENGTH, 1)
    cnn_model.fit(X_train_tf, y_train_tf, epochs=20, batch_size=16, verbose=0)
    cnn_pred = cnn_model.predict(X_test_tf).flatten()
    mae, rmse, mda = evaluate_metrics(y_test_tf.flatten(), cnn_pred)
    results['CNN'] = (mae, rmse, mda)
    print('CNN done.')
    
    hybrid_model = build_hybrid_cnn_lstm(SEQ_LENGTH, 1)
    hybrid_model.fit(X_train_tf, y_train_tf, epochs=20, batch_size=16, verbose=0)
    hybrid_pred = hybrid_model.predict(X_test_tf).flatten()
    mae, rmse, mda = evaluate_metrics(y_test_tf.flatten(), hybrid_pred)
    results['Hybrid (CNN-LSTM)'] = (mae, rmse, mda)
    print('Hybrid (CNN-LSTM) done.')
    
    sdtp_model = build_sdtp(SEQ_LENGTH, 1)
    sdtp_model.fit(X_train_tf, y_train_tf, epochs=20, batch_size=16, verbose=0)
    sdtp_pred = sdtp_model.predict(X_test_tf).flatten()
    mae, rmse, mda = evaluate_metrics(y_test_tf.flatten(), sdtp_pred)
    results['SDTP'] = (mae, rmse, mda)
    print('SDTP done.')
    
    numhtml_model = build_numhtml(SEQ_LENGTH, 1)
    numhtml_model.fit(X_train_tf, y_train_tf, epochs=20, batch_size=16, verbose=0)
    numhtml_pred = numhtml_model.predict(X_test_tf).flatten()
    mae, rmse, mda = evaluate_metrics(y_test_tf.flatten(), numhtml_pred)
    results['NumHTML'] = (mae, rmse, mda)
    print('NumHTML done.')
    
    results_df = pd.DataFrame(results, index=['MAE', 'RMSE', 'MDA (%)']).T
    print('\nModel Comparison:')
    print(results_df)

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_tf, label='True')
    plt.plot(lstm_pred, label='LSTM Prediction')
    plt.plot(cnn_pred, label='CNN Prediction')
    plt.plot(hybrid_pred, label='Hybrid Prediction')
    plt.plot(sdtp_pred, label='SDTP Prediction')
    plt.plot(numhtml_pred, label='NumHTML Prediction')
    plt.legend()
    plt.title('Model Predictions on S&P500')
    plt.show()