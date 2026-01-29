# PyQuantumForecaster

Hybrid quantum-classical engine for predicting oscillatory time series.

## Features
- Variational quantum circuits for capturing periodic patterns
- Simple training and multi-step prediction
- Works on simulators (no real quantum hardware needed)

## Installation
```bash
pip install -e .   # from source
# Or later:
pip install pyquantumforecaster
```

## Quick Start (Sine Example)

```python
import numpy as np
from pyquantumforecaster import QuantumForecaster, generate_sine_data, create_windows, train_model, predict_next

# Generate data
data = generate_sine_data(length=1000)
X, y = create_windows(data, window_size=50)

# Train
model = QuantumForecaster(window_size=50, n_qubits=6)
losses = train_model(model, X, y, epochs=200)

# Predict next 10 steps from last window
last_window = data[-50:]
preds = predict_next(model, last_window, steps=10)
print("Predicted next 10:", preds)
```

## Notes
- This project uses PennyLane for variational quantum circuits and PyTorch for classical layers.
- The included examples run on simulators; no quantum hardware is required to try the library.
