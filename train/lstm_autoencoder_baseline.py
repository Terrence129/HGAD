import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMAutoencoder(nn.Module):
    """
    Input shape: (batch_size, window_size, num_features)
    Output shape: (batch_size, window_size, num_features)
    """

    def __init__(
        self,
        num_features: int = 38,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_layers == 1 and dropout > 0:
            dropout = 0.0

        self.num_features = num_features
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.output_layer = nn.Linear(hidden_size, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (B, T, F), got shape={tuple(x.shape)}")
        if x.size(-1) != self.num_features:
            raise ValueError(
                f"Expected feature dim={self.num_features}, got {x.size(-1)}"
            )

        batch_size, seq_len, _ = x.shape

        _, (h_n, c_n) = self.encoder(x)

        # Use zero decoder inputs and encoder states to reconstruct the full sequence.
        decoder_input = torch.zeros(
            batch_size, seq_len, self.hidden_size, device=x.device, dtype=x.dtype
        )
        decoder_output, _ = self.decoder(decoder_input, (h_n, c_n))
        x_hat = self.output_layer(decoder_output)

        return x_hat


def train(
    model: nn.Module,
    train_windows: np.ndarray | torch.Tensor,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    shuffle: bool = True,
) -> list[float]:
    """
    Train LSTM Autoencoder with MSE loss + Adam optimizer.
    train_windows shape: (N, window_size, num_features)
    """

    model = model.to(device)
    model.train()

    if isinstance(train_windows, np.ndarray):
        x_tensor = torch.tensor(train_windows, dtype=torch.float32)
    else:
        x_tensor = train_windows.float()

    if x_tensor.ndim != 3:
        raise ValueError(
            f"Expected train_windows shape (N, T, F), got {tuple(x_tensor.shape)}"
        )

    loader = DataLoader(
        TensorDataset(x_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)

            batch_recon = model(batch_x)
            loss = criterion(batch_recon, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(loader.dataset)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch:03d}/{epochs:03d} | loss={avg_loss:.6f}")

    return epoch_losses


@torch.no_grad()
def compute_scores(
    model: nn.Module,
    windows: np.ndarray | torch.Tensor,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """
    Compute anomaly score for each window.
    anomaly score = reconstruction MSE over (time, feature)
    Return shape: (N,)
    """

    model = model.to(device)
    model.eval()

    if isinstance(windows, np.ndarray):
        x_tensor = torch.tensor(windows, dtype=torch.float32)
    else:
        x_tensor = windows.float()

    if x_tensor.ndim != 3:
        raise ValueError(f"Expected windows shape (N, T, F), got {tuple(x_tensor.shape)}")

    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=False)

    all_scores = []
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        batch_recon = model(batch_x)

        batch_scores = torch.mean((batch_recon - batch_x) ** 2, dim=(1, 2))
        all_scores.append(batch_scores.cpu())

    return torch.cat(all_scores, dim=0).numpy()

