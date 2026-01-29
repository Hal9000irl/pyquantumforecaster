import pennylane as qml
import torch
import torch.nn as nn
from typing import Optional

# Create a global device for the quantum circuit
dev = qml.device("default.qubit", wires=8)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights, n_qubits):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))  # predict next scalar

class QuantumForecaster(nn.Module):
    def __init__(self, window_size: int = 50, n_qubits: int = 8, n_layers: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        # Update global device to match n_qubits
        global dev
        dev = qml.device("default.qubit", wires=n_qubits)
        
        self.classical_pre = nn.Linear(window_size, n_qubits)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.q_weights = nn.Parameter(torch.randn(shape, dtype=torch.float32) * 0.01)
        self.q_weights.data = self.q_weights.data.to(torch.float32)
        # Ensure entire model is float32
        self.to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, window_size)
        x = self.classical_pre(x)  # → (batch, n_qubits)
        preds = []
        for sample in x:
            pred = quantum_circuit(sample, self.q_weights, self.n_qubits)
            preds.append(pred)
        return torch.stack(preds)

import numpy as np
from torch.optim import Adam
from torch.nn import MSELoss

def train_model(model: QuantumForecaster, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 300, lr: float = 0.01):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = MSELoss()
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    model.train()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_tensor)
        preds = preds.to(torch.float32)  # Ensure float32 for loss computation
        loss = criterion(preds.unsqueeze(1), y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    return losses

def predict_next(model: QuantumForecaster, window: np.ndarray, steps: int = 10):
    model.eval()
    predictions = []
    current_window = window.copy().astype(np.float32)
    
    for _ in range(steps):
        input_tensor = torch.tensor(current_window.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            next_val = model(input_tensor).item()
        predictions.append(next_val)
        current_window = np.append(current_window[1:], next_val)
    
    return np.array(predictions)