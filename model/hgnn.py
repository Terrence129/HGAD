import torch
import torch.nn as nn


class HGNNLayer(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, H):

        X = X.float()
        H = H.float()

        # node degree
        Dv = torch.sum(H, dim=1)
        Dv_inv_sqrt = torch.diag(torch.pow(Dv + 1e-6, -0.5))

        # edge degree
        De = torch.sum(H, dim=0)
        De_inv = torch.diag(1.0 / (De + 1e-6))

        # normalized propagation
        H_norm = Dv_inv_sqrt @ H @ De_inv @ H.t() @ Dv_inv_sqrt

        X = H_norm @ X

        return self.linear(X)


class HGNN_AE(nn.Module):

    def __init__(self, in_dim=10, hidden_dim=32, embed_dim=16):
        super().__init__()

        # encoder
        self.enc1 = HGNNLayer(in_dim, hidden_dim)
        self.enc2 = HGNNLayer(hidden_dim, embed_dim)

        # decoder
        self.dec1 = HGNNLayer(embed_dim, hidden_dim)
        self.dec2 = HGNNLayer(hidden_dim, in_dim)

        self.relu = nn.ReLU()

    def forward(self, X, H):

        # encode
        Z = self.enc1(X, H)
        Z = self.relu(Z)

        Z = self.enc2(Z, H)

        # decode
        X_hat = self.dec1(Z, H)
        X_hat = self.relu(X_hat)

        X_hat = self.dec2(X_hat, H)

        return X_hat, Z