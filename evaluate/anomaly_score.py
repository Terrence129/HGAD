import numpy as np
import torch
import torch.nn.functional as F
from hypergraph.build_hypergraph import build_incidence_from_window

def compute_scores(model, W: np.ndarray, k: int = 5, tau: float = 0.7) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for Xw_np in W.astype(np.float32):
            H_np = build_incidence_from_window(Xw_np, k=k, tau=tau)
            Xw = torch.tensor(Xw_np)
            H = torch.tensor(H_np)
            Xhat, _ = model(Xw, H)
            mse = F.mse_loss(Xhat, Xw, reduction="mean").item()
            scores.append(mse)
    return np.array(scores)