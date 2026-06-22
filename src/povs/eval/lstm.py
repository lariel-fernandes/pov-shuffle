from typing import NamedTuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMLayerOptions(TypedDict):
    hidden_size: int


class LSTMSettings(NamedTuple):
    layers: list[int] | list[LSTMLayerOptions]
    context_length: int = 16
    learning_rate: float = 1e-3
    num_epochs: int = 10
    batch_size: int = 32
    train_ratio: float = 0.8
    max_sequences: int | None = None


def lstm_predictability(
    samples: np.ndarray,
    deck_size: int,
    settings: LSTMSettings,
    device: str,
) -> float:
    """LSTM-based predictability of n-gram RTD sequences, normalized to [0, 1].

    Returns ``1 - test_mse / naive_mse`` where ``naive_mse`` is the variance of the test targets
    (MSE of always predicting the test-set mean), making this equivalent to the coefficient of
    determination (R²) on the test split. A value of 0 means the LSTM performs no better than
    predicting the mean (indistinguishable from a uniform shuffle); 1 means perfect prediction.

    :param samples: Array of shape ``(num_samples, deck_size)`` containing independent permutations.
    :param deck_size: Number of elements in the deck; used for RTD normalization.
    :param settings: LSTM architecture and training hyperparameters.
    :param device: Torch device for model and tensors (e.g. ``"cpu"``, ``"cuda"``).
    """
    sequences = _prepare_sequences(samples, deck_size, settings.context_length, settings.max_sequences)
    test_mse, naive_mse = _train_and_eval(sequences, settings, device)
    return 1.0 - test_mse / max(naive_mse, 1e-8)


def _prepare_sequences(samples: np.ndarray, deck_size: int, n: int, max_sequences: int | None = None) -> torch.Tensor:
    """Extract n-gram RTD sequences from permutation samples.

    For each n-gram of n consecutive elements (with wrap-around), computes n-1 signed RTDs
    normalized to ``[-1, 1]``. The first n-2 RTDs serve as context; the last is the prediction target.

    If ``max_sequences`` is set, a random subset of deck positions is drawn per sample *before* building
    the RTD array, keeping peak memory proportional to ``max_sequences`` rather than
    ``num_samples * deck_size``.

    :returns: Float32 tensor of shape ``(min(num_samples * deck_size, max_sequences), n-1)``.
    """
    num_samples = len(samples)
    if max_sequences is not None and num_samples * deck_size > max_sequences:
        positions_per_sample = max(1, max_sequences // num_samples)
        pos = np.random.choice(deck_size, positions_per_sample, replace=False)
    else:
        positions_per_sample = deck_size
        pos = np.arange(deck_size)

    col_indices = (pos.reshape(-1, 1) + np.arange(n).reshape(1, n)) % deck_size
    ngrams = samples[:, col_indices]  # (num_samples, positions_per_sample, n)
    expected = np.arange(1, n, dtype=np.int64)
    diffs = ngrams[..., 1:].astype(np.int64) - ngrams[..., :1].astype(np.int64)
    rtds = (diffs - expected) / deck_size  # (num_samples, positions_per_sample, n-1)
    return torch.tensor(rtds.reshape(-1, n - 1), dtype=torch.float32)


def _train_and_eval(sequences: torch.Tensor, settings: LSTMSettings, device: str) -> tuple[float, float]:
    """Train an LSTM to predict the last RTD given prior RTDs; return ``(test_mse, naive_mse)``.

    Each n-gram is one example: the first n-2 RTDs are fed as a scalar sequence, the last RTD is
    the target. ``naive_mse`` is the MSE of always predicting the test-set mean (lower bound for
    a model that cannot exploit sequential structure).
    """
    x = sequences[:, :-1].unsqueeze(-1)  # (total, n-2, 1)
    y = sequences[:, -1:]  # (total, 1)

    split = int(len(sequences) * settings.train_ratio)
    layer_sizes = [lyr["hidden_size"] if isinstance(lyr, dict) else lyr for lyr in settings.layers]

    model = _LSTMModel(1, layer_sizes, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(x[:split], y[:split]),
        batch_size=settings.batch_size,
        shuffle=True,
    )
    model.train()
    for _ in range(settings.num_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb.to(device)), yb.to(device)).backward()
            optimizer.step()

    y_test = y[split:]
    model.eval()
    with torch.no_grad():
        eval_loader = DataLoader(TensorDataset(x[split:]), batch_size=settings.batch_size)
        preds = torch.cat([model(xb.to(device)).cpu() for (xb,) in eval_loader])
    test_mse = loss_fn(preds, y_test).item()
    naive_mse = (y_test - y_test.mean()).pow(2).mean().item()
    return test_mse, naive_mse


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
        return self.head(out[:, -1, :])  # seq2one: only last time step feeds the head
