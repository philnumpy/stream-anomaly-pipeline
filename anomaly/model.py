"""LSTM autoencoder for multivariate time-series anomaly detection.

The encoder compresses a window of readings into one hidden vector; the
decoder has to rebuild the window from that vector alone. Trained only on
normal data, it rebuilds normal windows well and unusual ones badly, so the
reconstruction error is the anomaly score.
"""

import numpy as np
import torch
from torch import nn

WINDOW = 50      # readings per window (50 minutes of history in SMD)
N_FEATURES = 38  # metrics per reading in SMD
HIDDEN = 64


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden: int = HIDDEN):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, window, features]
        _, (h, _) = self.encoder(x)
        # Feed the same summary vector to the decoder at every time step.
        summary = h[-1].unsqueeze(1).expand(-1, x.shape[1], -1)
        decoded, _ = self.decoder(summary)
        return self.output(decoded)


class LastStepReconstruction(nn.Module):
    """Export wrapper: returns only the rebuilt newest reading, [batch, features].

    The anomaly score of a reading is the error on the last step of the window
    that ends at it, so the serving side never needs the rest.
    """

    def __init__(self, model: LSTMAutoencoder):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[:, -1, :]


def preprocess(x: np.ndarray) -> np.ndarray:
    """SMD is already scaled to 0-1 using the training range, but test values
    can fall far outside it. Clipping keeps one broken metric from dominating.
    Used identically offline and in the stream worker."""
    return np.clip(x, -1.0, 2.0).astype(np.float32)


def make_windows(x: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """[T, F] -> [T - window + 1, window, F] as a view (no copy)."""
    return np.lib.stride_tricks.sliding_window_view(x, window, axis=0).transpose(0, 2, 1)
