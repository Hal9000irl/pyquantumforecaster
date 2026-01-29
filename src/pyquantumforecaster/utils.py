import numpy as np
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_crypto_data(filepath: str, column: str = 'Close', normalize: bool = True):
    """
    Load OHLCV CSV (e.g., from Yahoo Finance or CoinGecko export).
    Returns: numpy array of selected column (normalized 0-1 if normalize=True)
    """
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    series = df[column].dropna().values.astype(np.float32)
    
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        series = scaler.fit_transform(series.reshape(-1, 1)).flatten()
        return series, scaler  # return data + fitted scaler for denormalizing later
    return series, None
def generate_sine_data(length: int = 1000, frequency: float = 0.1, noise: float = 0.0) -> np.ndarray:
    """
    Generate synthetic sine wave data for testing.
    
    Args:
        length: Number of data points
        frequency: Frequency of the sine wave
        noise: Standard deviation of Gaussian noise to add
        
    Returns:
        1D numpy array of sine wave values
    """
    t = np.arange(length)
    data = np.sin(2 * np.pi * frequency * t)
    if noise > 0:
        data += np.random.normal(0, noise, length)
    return data.astype(np.float32)

def create_windows(data: np.ndarray, window_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series prediction.
    
    Args:
        data: 1D array of time series values
        window_size: Size of each window
        
    Returns:
        Tuple of (X, y) where X is (n_samples, window_size) and y is (n_samples,)
    """
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

import matplotlib.pyplot as plt

def plot_predictions(actual: np.ndarray, predicted: np.ndarray, title: str = "Actual vs Predicted Oscillations"):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual', color='blue', linewidth=2)
    plt.plot(range(len(actual)-len(predicted), len(actual)), predicted, 
             label='Predicted', color='orange', linestyle='--', linewidth=2.5)
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Value (or USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        # Headless environment: save to file
        plt.savefig("predictions_plot.png")
