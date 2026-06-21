from typing import NamedTuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMLayerOptions(TypedDict):
    hidden_size: int


class LSTMSettings(NamedTuple):
    layers: list[int] | list[LSTMLayerOptions]
    learning_rate: float = 1e-3
    num_epochs: int = 10
    batch_size: int = 32
    train_ratio: float = 0.8


def lstm_predictability(
    samples: np.ndarray,
    deck_size: int,
    n: int,
    settings: LSTMSettings,
    device: str,
) -> float:
    """LSTM-based predictability of n-gram RTD sequences, normalized to [0, 1].

    Returns ``1 - test_mse / 4`` where 4 is the max MSE for outputs in ``[-1, 1]``.
    A value near 0 means unpredictable RTD sequences (good shuffler);
    near 1 means predictable (biased shuffler).

    :param samples: Array of shape ``(num_samples, deck_size)`` containing independent permutations.
    :param deck_size: Number of elements in the deck; used for RTD normalization.
    :param n: N-gram degree; determines the RTD feature size ``n-1`` per time step.
    :param settings: LSTM architecture and training hyperparameters.
    :param device: Torch device for model and tensors (e.g. ``"cpu"``, ``"cuda"``).
    """
    sequences = _prepare_sequences(samples, deck_size, n)
    test_mse = _train_and_eval(sequences, settings, device)
    return 1.0 - test_mse / 4.0


def _prepare_sequences(samples: np.ndarray, deck_size: int, n: int) -> torch.Tensor:
    """Extract signed, normalized RTD sequences from permutation samples.

    Uses adjacent n-grams (skip=0). RTD formula: ``rtd_k = (v_k - v_0) - k``,
    normalized by ``deck_size`` to ``[-1, 1]``.

    :returns: Float32 tensor of shape ``(num_samples, deck_size, n-1)``.
    """
    col_indices = np.arange(deck_size).reshape(deck_size, 1) + np.arange(n).reshape(1, n)
    ngrams = samples.take(col_indices, axis=1, mode="wrap")  # (num_samples, deck_size, n)
    expected = np.arange(1, n, dtype=np.int64)
    diffs = ngrams[..., 1:].astype(np.int64) - ngrams[..., :1].astype(np.int64)
    rtds = (diffs - expected) / deck_size
    return torch.tensor(rtds, dtype=torch.float32)


def _train_and_eval(sequences: torch.Tensor, settings: LSTMSettings, device: str) -> float:
    """Train an LSTM to predict the next RTD vector and return test MSE."""
    x = sequences[:, :-1, :]  # inputs: positions 0..deck_size-2
    y = sequences[:, 1:, :]   # targets: positions 1..deck_size-1

    split = int(len(sequences) * settings.train_ratio)
    feat_dim = sequences.shape[-1]
    layer_sizes = [lyr["hidden_size"] if isinstance(lyr, dict) else lyr for lyr in settings.layers]

    model = _LSTMModel(feat_dim, layer_sizes, feat_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(x[:split].to(device), y[:split].to(device)),
        batch_size=settings.batch_size,
        shuffle=True,
    )
    model.train()
    for _ in range(settings.num_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_mse = loss_fn(model(x[split:].to(device)), y[split:].to(device)).item()
    return test_mse


class _LSTMModel(nn.Module):
    def __init__(self, input_size: int, layer_hidden_sizes: list[int], output_size: int):
        super().__init__()
        self.lstms = nn.ModuleList()
        in_size = input_size
        for hidden_size in layer_hidden_sizes:
            self.lstms.append(nn.LSTM(in_size, hidden_size, batch_first=True))
            in_size = hidden_size
        self.head = nn.Sequential(nn.Linear(in_size, output_size), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for lstm in self.lstms:
            out, _ = lstm(out)
        return self.head(out)
