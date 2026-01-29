import numpy as np
from pyquantumforecaster import QuantumForecaster, generate_sine_data, create_windows, train_model, predict_next

print("Generating data...")
data = generate_sine_data(length=300)  # smaller for quick test
X, y = create_windows(data, window_size=50)

print("Initializing model...")
model = QuantumForecaster(window_size=50, n_qubits=4)  # small qubits for speed

print("Training (short run)...")
losses = train_model(model, X, y, epochs=100, lr=0.02)  # short epochs

print("Predicting next 5 steps from last window...")
last_window = data[-50:]
preds = predict_next(model, last_window, steps=5)
print("Predictions:", preds)
