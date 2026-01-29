import numpy as np
from pyquantumforecaster import (
    QuantumForecaster, create_windows, train_model, predict_next,
    load_crypto_data, plot_predictions
)

# Load data (use your btc_sample.csv or fallback)
print("Loading data...")
try:
    data, scaler = load_crypto_data("examples/btc_sample.csv")
except FileNotFoundError:
    print("No CSV → fallback to synthetic")
    from pyquantumforecaster.utils import generate_sine_data
    data = generate_sine_data(length=400)
    scaler = None

print("Data shape:", data.shape)

# Train on first ~80%
split = int(len(data) * 0.8)
train_data = data[:split]
test_data = data[split:]

X_train, y_train = create_windows(train_data, window_size=50)
model = QuantumForecaster(window_size=50, n_qubits=6)

print("Training...")
losses = train_model(model, X_train, y_train, epochs=80, lr=0.015)

# Predict on test window (last train window → forecast test length)
print("Forecasting test period...")
last_train_window = train_data[-50:]
preds = predict_next(model, last_train_window, steps=len(test_data))

# Denormalize if scaler exists
if scaler is not None:
    test_data = scaler.inverse_transform(test_data.reshape(-1, 1)).flatten()
    preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    title = "BTC-USD: Actual vs Predicted (denormalized)"
else:
    title = "Synthetic Oscillations: Actual vs Predicted"

# Plot
full_actual = np.concatenate([train_data[-100:], test_data])  # show last bit of train + test
plot_predictions(full_actual, preds, title=title)
print("Plot should appear in popup/window or be saved to predictions_plot.png.")
