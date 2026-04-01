import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from hypergraph.build_hypergraph import (
    build_local_incidence_batch_torch,
    combine_global_local_incidence,
)


def build_hybrid_incidence(
    x_nodes: torch.Tensor,
    h_global_t: torch.Tensor,
    graph_mode: str = "global",
    local_top_k: int = 4,
) -> torch.Tensor:
    if graph_mode == "global":
        return h_global_t.unsqueeze(0).expand(x_nodes.shape[0], -1, -1)
    if graph_mode == "global_local":
        h_local = build_local_incidence_batch_torch(x_nodes, top_k=local_top_k)
        return combine_global_local_incidence(h_global_t, h_local)
    raise ValueError(f"Unsupported graph_mode={graph_mode}, expected 'global' or 'global_local'")


def moving_average(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, np.ones(k) / k, mode="valid")


@torch.no_grad()
def compute_embedding_stats(
    model,
    windows: np.ndarray,
    h_global: np.ndarray,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    local_top_k: int = 4,
    graph_mode: str = "global",
) -> dict[str, np.ndarray]:
    model.eval()
    device = torch.device(device)

    w = torch.tensor(windows, dtype=torch.float32)
    hg = torch.tensor(h_global, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(w), batch_size=batch_size, shuffle=False)

    pooled = []
    for (batch_w,) in loader:
        batch_w = batch_w.to(device)  # (B, T, F)
        x_nodes = batch_w.transpose(1, 2)  # (B, F, T)
        h_batch = build_hybrid_incidence(
            x_nodes=x_nodes,
            h_global_t=hg,
            graph_mode=graph_mode,
            local_top_k=local_top_k,
        )
        _, z, _ = model(x_nodes, h_batch)
        pooled.append(torch.mean(z, dim=1).cpu())

    z_all = torch.cat(pooled, dim=0).numpy()
    mu = z_all.mean(axis=0)
    std = z_all.std(axis=0) + 1e-6
    return {"mu": mu, "std": std}


@torch.no_grad()
def compute_anomaly_scores(
    model,
    windows: np.ndarray,
    h_global: np.ndarray,
    emb_stats: dict[str, np.ndarray] | None = None,
    score_weights: tuple[float, float, float] = (0.6, 0.25, 0.15),
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    local_top_k: int = 4,
    graph_mode: str = "global",
) -> dict[str, np.ndarray]:
    """
    windows: (Nw, T, F)
    h_global: (F, Eg)
    """
    model.eval()
    device = torch.device(device)

    w = torch.tensor(windows, dtype=torch.float32)
    hg = torch.tensor(h_global, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(w), batch_size=batch_size, shuffle=False)

    w_recon, w_temp, w_embed, w_total = [], [], [], []
    n_recon, n_temp = [], []

    for (batch_w,) in loader:
        batch_w = batch_w.to(device)  # (B, T, F)
        x_nodes = batch_w.transpose(1, 2)  # (B, F, T)

        h_batch = build_hybrid_incidence(
            x_nodes=x_nodes,
            h_global_t=hg,
            graph_mode=graph_mode,
            local_top_k=local_top_k,
        )

        x_hat, z, _ = model(x_nodes, h_batch)

        node_recon = torch.mean((x_hat - x_nodes) ** 2, dim=2)  # (B, F)
        win_recon = node_recon.mean(dim=1)  # (B,)

        dx = torch.diff(x_nodes, dim=2)
        dx_hat = torch.diff(x_hat, dim=2)
        node_temp = torch.mean((dx_hat - dx) ** 2, dim=2)  # (B, F)
        win_temp = node_temp.mean(dim=1)  # (B,)

        if emb_stats is not None:
            z_pool = torch.mean(z, dim=1)  # (B, D)
            mu = torch.tensor(emb_stats["mu"], dtype=torch.float32, device=device)
            std = torch.tensor(emb_stats["std"], dtype=torch.float32, device=device)
            win_embed = torch.mean(((z_pool - mu) / std) ** 2, dim=1)
        else:
            win_embed = torch.zeros_like(win_recon)

        a, b, c = score_weights
        win_total = a * win_recon + b * win_temp + c * win_embed

        w_recon.append(win_recon.cpu())
        w_temp.append(win_temp.cpu())
        w_embed.append(win_embed.cpu())
        w_total.append(win_total.cpu())
        n_recon.append(node_recon.cpu())
        n_temp.append(node_temp.cpu())

    return {
        "window_recon": torch.cat(w_recon).numpy(),
        "window_temporal": torch.cat(w_temp).numpy(),
        "window_embedding": torch.cat(w_embed).numpy(),
        "window_total": torch.cat(w_total).numpy(),
        "node_recon": torch.cat(n_recon).numpy(),   # (Nw, F)
        "node_temporal": torch.cat(n_temp).numpy(),  # (Nw, F)
    }


def search_threshold_by_percentile(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    y_test: np.ndarray,
    percentiles: list[float],
) -> dict:
    best = None
    for q in percentiles:
        threshold = np.percentile(train_scores, q)
        pred = (test_scores > threshold).astype(np.int64)

        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)

        result = {
            "percentile": q,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pred": pred,
        }
        if best is None or result["f1"] > best["f1"]:
            best = result
    return best


def compute_root_cause_scores(
    node_recon: np.ndarray,
    node_temporal: np.ndarray,
    h_incidence: np.ndarray,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> np.ndarray:
    """
    node_recon: (F,)
    node_temporal: (F,)
    h_incidence: (F, E)
    return node_scores: (F,)
    """
    eps = 1e-6
    h = h_incidence.astype(np.float32)
    node_r = node_recon.astype(np.float32)
    node_t = node_temporal.astype(np.float32)

    de = np.sum(h, axis=0) + eps
    edge_energy = (h.T @ node_r) / de

    dv = np.sum(h, axis=1) + eps
    propagated = (h @ edge_energy) / dv

    def _z(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-6)

    node_r_n = _z(node_r)
    node_t_n = _z(node_t)
    prop_n = _z(propagated)

    a, b, c = weights
    return a * node_r_n + b * node_t_n + c * prop_n


def topk_root_causes_for_window(
    node_recon: np.ndarray,
    node_temporal: np.ndarray,
    h_incidence: np.ndarray,
    metric_names: list[str],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    scores = compute_root_cause_scores(node_recon, node_temporal, h_incidence)
    idx = np.argsort(scores)[::-1][:top_k]
    return [(metric_names[i], float(scores[i])) for i in idx]


def global_root_cause_ranking(
    all_node_recon: np.ndarray,
    all_node_temporal: np.ndarray,
    pred: np.ndarray,
    h_global: np.ndarray,
    metric_names: list[str],
    top_k_each: int = 3,
) -> list[tuple[str, int]]:
    counts = {name: 0 for name in metric_names}
    anomaly_idx = np.where(pred == 1)[0]

    if len(anomaly_idx) == 0:
        return []

    for i in anomaly_idx:
        topk = topk_root_causes_for_window(
            all_node_recon[i],
            all_node_temporal[i],
            h_global,
            metric_names=metric_names,
            top_k=top_k_each,
        )
        for name, _ in topk:
            counts[name] += 1

    ranking = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return ranking
