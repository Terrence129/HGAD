import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    x: (B, N, Fin) or (N, Fin)
    a_hat: (N, N) or (B, N, N)
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        squeeze_back = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze_back = True
        if a_hat.ndim == 2:
            a_hat = a_hat.unsqueeze(0).expand(x.shape[0], -1, -1)

        if x.ndim != 3 or a_hat.ndim != 3:
            raise ValueError(
                f"Expected x=(B,N,F), a_hat=(B,N,N), got x={tuple(x.shape)}, a={tuple(a_hat.shape)}"
            )

        x = x.float()
        a_hat = a_hat.float()

        out = torch.bmm(a_hat, x)
        out = self.linear(out)

        if squeeze_back:
            out = out.squeeze(0)
        return out


class GCNAutoencoder(nn.Module):
    """
    Input:
      x: (B, N, in_dim) or (N, in_dim)
      a_hat: (N, N) or (B, N, N)
    Output:
      x_hat: same shape as x
      z: latent embedding
    """

    def __init__(
        self,
        in_dim: int = 10,
        hidden_dim: int = 32,
        embed_dim: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.enc1 = GCNLayer(in_dim, hidden_dim)
        self.enc2 = GCNLayer(hidden_dim, embed_dim)
        self.dec1 = GCNLayer(embed_dim, hidden_dim)
        self.dec2 = GCNLayer(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze_back = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze_back = True

        z = self.enc1(x, a_hat)
        z = F.relu(z)
        z = self.dropout(z)
        z = self.enc2(z, a_hat)

        x_hat = self.dec1(z, a_hat)
        x_hat = F.relu(x_hat)
        x_hat = self.dropout(x_hat)
        x_hat = self.dec2(x_hat, a_hat)

        if squeeze_back:
            x_hat = x_hat.squeeze(0)
            z = z.squeeze(0)
        return x_hat, z
