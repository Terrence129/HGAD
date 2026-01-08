import torch
import torch.nn as nn

class SimpleHypergraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # x: (d, in_dim)  nodes are features
        # H: (d, m)
        x = self.lin(x)
        he = H.T @ x          # (m, out_dim)
        x2 = H @ he           # (d, out_dim)
        return x2

class HGADDetector(nn.Module):
    def __init__(self, window: int, d: int, hidden: int = 64, emb: int = 32):
        super().__init__()
        self.window = window
        self.d = d
        self.enc = nn.Sequential(
            nn.Linear(window, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb),
        )
        self.hconv1 = SimpleHypergraphConv(emb, emb)
        self.hconv2 = SimpleHypergraphConv(emb, emb)
        self.dec = nn.Sequential(
            nn.Linear(emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, window),
        )

    def forward(self, Xw: torch.Tensor, H: torch.Tensor):
        # Xw: (window, d) -> treat each metric as node with window-length feature
        Xw = Xw.T  # (d, window)
        z = self.enc(Xw)  # (d, emb)
        z = torch.relu(self.hconv1(z, H))
        z = torch.relu(self.hconv2(z, H))
        Xhat = self.dec(z)  # (d, window)
        Xhat = Xhat.T  # (window, d)
        return Xhat, z