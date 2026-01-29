import pennylane as qml
import torch
import torch.nn as nn
from typing import Optional

class QuantumForecaster(nn.Module):
    def __init__(self, window_size: int = 50, n_qubits: int = 8, n_layers: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        
        self.classical_pre = nn.Linear(window_size, n_qubits)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.q_weights = nn.Parameter(torch.randn(shape) * 0.01)

    @qml.qnode(device="lightning.qubit", interface="torch")
    def circuit(self, inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        return qml.expval(qml.PauliZ(0))  # predict next scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, window_size)
        x = self.classical_pre(x)  # → (batch, n_qubits)
        preds = []
        for sample in x:
            pred = self.circuit(sample, self.q_weights)
            preds.append(pred)
        return torch.stack(preds)
