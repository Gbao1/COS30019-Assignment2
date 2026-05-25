import torch
import torch.nn as nn

class LSTMTrafficModel(nn.Module):
    """Long Short-Term Memory network architecture for sequential volume forecasting."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Extract last step output vector


class GRUTrafficModel(nn.Module):
    """Gated Recurrent Unit model acting as our primary alternative network benchmark."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class MLPPredictor(nn.Module):
    """A standard Multi-Layer Perceptron used as the third algorithm for the report evaluation."""
    def __init__(self, input_size=4, hidden_size=64, output_size=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # Flatten sequence input dimension for classical feed-forward pass
        x = x.squeeze(-1)
        return self.network(x)
