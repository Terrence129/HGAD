import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveHypergraphConv(nn.Module):
    """
    Hypergraph convolution with per-sample adaptive edge gates.

    x: (B, N, F_in)
    h: (B, N, E)
    out: (B, N, F_out)
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.edge_gate = nn.Sequential(
            nn.Linear(in_dim, max(in_dim // 2, 8)),
            nn.GELU(),
            nn.Linear(max(in_dim // 2, 8), 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must be 3D (B, N, F), got {tuple(x.shape)}")
        if h.ndim != 3:
            raise ValueError(f"h must be 3D (B, N, E), got {tuple(h.shape)}")
        if x.shape[0] != h.shape[0] or x.shape[1] != h.shape[1]:
            raise ValueError(f"x shape {tuple(x.shape)} incompatible with h {tuple(h.shape)}")

        x = x.float()
        h = h.float()

        eps = 1e-6
        de = torch.sum(h, dim=1, keepdim=False)  # (B, E)
        edge_seed = torch.bmm(h.transpose(1, 2), x) / (de.unsqueeze(-1) + eps)  # (B, E, F)
        gate = torch.sigmoid(self.edge_gate(edge_seed)).squeeze(-1)  # (B, E)

        h_w = h * gate.unsqueeze(1)  # (B, N, E)

        dv = torch.sum(h_w, dim=2)  # (B, N)
        de_w = torch.sum(h_w, dim=1)  # (B, E)

        dv_inv_sqrt = torch.pow(dv + eps, -0.5).unsqueeze(-1)  # (B, N, 1)
        de_inv = torch.pow(de_w + eps, -1.0).unsqueeze(-1)  # (B, E, 1)

        x_norm = x * dv_inv_sqrt
        edge_feat = torch.bmm(h_w.transpose(1, 2), x_norm) * de_inv  # (B, E, F)
        node_feat = torch.bmm(h_w, edge_feat) * dv_inv_sqrt  # (B, N, F)

        out = self.linear(node_feat)
        out = self.dropout(out)
        out = out + self.res_proj(x)
        return out, gate


class HierarchicalAdaptiveHGNN_AE(nn.Module):
    """
    Hierarchical HGNN Autoencoder for node-feature reconstruction.

    Input:
      x: (N, in_dim) or (B, N, in_dim)
      h: (N, E) or (B, N, E)
    Output:
      x_hat: same shape as x
      z: latent embedding (N, embed_dim) or (B, N, embed_dim)
      details: dict with edge gates and intermediate embeddings
    """

    def __init__(
        self,
        in_dim: int = 10,
        hidden_dim: int = 32,
        embed_dim: int = 8,
        dropout: float = 0.2,
        latent_noise_std: float = 0.05,
        use_input_skip: bool = False,
    ):
        super().__init__()

        self.enc1 = AdaptiveHypergraphConv(in_dim, hidden_dim, dropout=dropout)
        self.enc2 = AdaptiveHypergraphConv(hidden_dim, embed_dim, dropout=dropout)

        self.dec1 = AdaptiveHypergraphConv(embed_dim, hidden_dim, dropout=dropout)
        self.dec2 = AdaptiveHypergraphConv(hidden_dim, in_dim, dropout=dropout)

        self.norm_e1 = nn.LayerNorm(hidden_dim)
        self.norm_e2 = nn.LayerNorm(embed_dim)
        self.norm_d1 = nn.LayerNorm(hidden_dim)
        self.norm_d2 = nn.LayerNorm(in_dim)

        self.latent_noise_std = latent_noise_std
        self.use_input_skip = use_input_skip
        self.input_skip = nn.Identity() if use_input_skip else None

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        squeeze_back = False

        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze_back = True
        if h.ndim == 2:
            h = h.unsqueeze(0).expand(x.shape[0], -1, -1)

        x0 = x.float()

        z1, g1 = self.enc1(x0, h)
        z1 = self.norm_e1(F.gelu(z1))

        z2, g2 = self.enc2(z1, h)
        z = self.norm_e2(F.gelu(z2))
        if self.training and self.latent_noise_std > 0:
            z = z + torch.randn_like(z) * self.latent_noise_std

        d1, g3 = self.dec1(z, h)
        d1 = self.norm_d1(F.gelu(d1))
        d1 = d1 + z1

        x_hat, g4 = self.dec2(d1, h)
        x_hat = self.norm_d2(x_hat)
        if self.input_skip is not None:
            x_hat = x_hat + self.input_skip(x0)

        details = {
            "gate_enc1": g1,
            "gate_enc2": g2,
            "gate_dec1": g3,
            "gate_dec2": g4,
            "z1": z1,
        }

        if squeeze_back:
            x_hat = x_hat.squeeze(0)
            z = z.squeeze(0)
            details = {
                k: (v.squeeze(0) if isinstance(v, torch.Tensor) and v.ndim >= 2 else v)
                for k, v in details.items()
            }

        return x_hat, z, details
