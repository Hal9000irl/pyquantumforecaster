import numpy as np
from pyquantumforecaster import QuantumForecaster, create_windows, train_model, predict_next, load_crypto_data

# You'll need a real CSV file - for now use synthetic as placeholder
print("Loading placeholder data (replace with real BTC CSV later)...")

# For testing: fall back to sine if no file
try:
    data, scaler = load_crypto_data("btc_sample.csv")
except FileNotFoundError:
    print("No CSV found → using synthetic sine as fallback")
    from pyquantumforecaster.utils import generate_sine_data
    data = generate_sine_data(length=300)
    scaler = None

print("Data shape:", data.shape)

X, y = create_windows(data, window_size=50)
model = QuantumForecaster(window_size=50, n_qubits=6)

print("Training...")
losses = train_model(model, X, y, epochs=80, lr=0.015)

print("Predicting next 10 steps...")
last_window = data[-50:]
preds = predict_next(model, last_window, steps=10)

if scaler is not None:
    preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    print("Denormalized predictions (approx USD):", preds)
else:
    print("Predictions (normalized):", preds)
