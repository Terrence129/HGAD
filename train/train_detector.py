import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from hypergraph.build_hypergraph import build_incidence_from_window
from model.hgnn_model import HGADDetector

class WindowDataset(Dataset):
    def __init__(self, W: np.ndarray):
        self.W = W.astype(np.float32)

    def __len__(self):
        return len(self.W)

    def __getitem__(self, idx):
        return self.W[idx]

def train_detector(W: np.ndarray, epochs: int = 3, lr: float = 1e-3, k: int = 5, tau: float = 0.7):
    device = "cpu"
    N, window, d = W.shape
    model = HGADDetector(window=window, d=d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(WindowDataset(W), batch_size=1, shuffle=True)

    model.train()
    for ep in range(1, epochs+1):
        total = 0.0
        for Xw_np in loader:
            Xw = Xw_np.squeeze(0).to(device)  # (window, d)
            H_np = build_incidence_from_window(Xw_np.squeeze(0).numpy(), k=k, tau=tau)
            H = torch.tensor(H_np, device=device)

            Xhat, _ = model(Xw, H)
            loss = F.mse_loss(Xhat, Xw)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
        print(f"Epoch {ep} | loss={total/len(loader):.6f}")

    return model