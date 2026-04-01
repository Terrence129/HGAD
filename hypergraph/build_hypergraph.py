from pathlib import Path
import numpy as np
import torch


def compute_correlation_matrix(X_train: np.ndarray) -> np.ndarray:
    """
    X_train: (T, F)
    return: (F, F)
    """
    if X_train.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={X_train.shape}")

    corr = np.corrcoef(X_train, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr


def build_incidence_from_correlation(
    corr: np.ndarray,
    threshold: float = 0.8,
    min_edge_size: int = 2,
    top_k: int = 6,
) -> np.ndarray:
    """
    corr: (F, F)
    return H: (F, E)
    """
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"corr must be square, got shape={corr.shape}")

    num_nodes = corr.shape[0]
    hyperedges = []

    for i in range(num_nodes):
        corr_i = np.abs(corr[i]).copy()

        # 不让自己参与排序干扰
        corr_i[i] = -1.0

        # 先按阈值筛
        candidate = np.where(corr_i >= threshold)[0]

        # 再按相关性从高到低取 top_k
        if len(candidate) > 0:
            candidate = candidate[np.argsort(corr_i[candidate])[::-1]]
            candidate = candidate[:top_k]

        # 把自己加回去
        group = np.unique(np.append(candidate, i))

        if len(group) >= min_edge_size:
            hyperedges.append(group)

    if len(hyperedges) == 0:
        raise ValueError(
            f"No hyperedges generated. Try lowering threshold={threshold}"
        )

    H = np.zeros((num_nodes, len(hyperedges)), dtype=np.float32)

    for e_idx, nodes in enumerate(hyperedges):
        H[nodes, e_idx] = 1.0

    return H


def window_to_node_features(window_x: np.ndarray) -> np.ndarray:
    """
    window_x: (window, F)
    return:   (F, window)
    """
    if window_x.ndim != 2:
        raise ValueError(f"Expected 2D window, got shape={window_x.shape}")

    return window_x.T.astype(np.float32)


def build_local_incidence_from_node_features(
    x_nodes: np.ndarray,
    top_k: int = 4,
) -> np.ndarray:
    """
    Build dynamic local hypergraph from a single window's node features.

    x_nodes: (N, T)
    return H_local: (N, N)
      - N local hyperedges
      - edge e_i contains node i + top_k most similar neighbors
    """
    if x_nodes.ndim != 2:
        raise ValueError(f"x_nodes must be 2D, got shape={x_nodes.shape}")

    num_nodes = x_nodes.shape[0]
    k = min(top_k, max(num_nodes - 1, 1))

    x = x_nodes.astype(np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    x_norm = x / norm
    sim = x_norm @ x_norm.T

    np.fill_diagonal(sim, -1.0)

    H_local = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        nbr = np.argsort(sim[i])[::-1][:k]
        H_local[i, i] = 1.0
        H_local[nbr, i] = 1.0

    return H_local


def build_local_incidence_batch_torch(
    x_nodes_batch: torch.Tensor,
    top_k: int = 4,
) -> torch.Tensor:
    """
    Torch batch version for dynamic local hypergraph construction.

    x_nodes_batch: (B, N, T)
    return H_local: (B, N, N)
    """
    if x_nodes_batch.ndim != 3:
        raise ValueError(
            f"x_nodes_batch must be 3D (B, N, T), got {tuple(x_nodes_batch.shape)}"
        )

    B, N, _ = x_nodes_batch.shape
    k = min(top_k, max(N - 1, 1))

    x = x_nodes_batch.float()
    x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-8)
    sim = torch.bmm(x, x.transpose(1, 2))

    eye = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0)
    sim = sim - 2.0 * eye

    nbr_idx = torch.topk(sim, k=k, dim=-1).indices  # (B, N, k)

    H_local = torch.zeros(B, N, N, device=x.device, dtype=x.dtype)

    anchor = torch.arange(N, device=x.device).view(1, N, 1).expand(B, N, 1)
    member_nodes = torch.cat([anchor, nbr_idx], dim=-1)  # (B, N, k+1)
    edge_ids = torch.arange(N, device=x.device).view(1, N, 1).expand(B, N, k + 1)
    batch_ids = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k + 1)

    H_local[batch_ids, member_nodes, edge_ids] = 1.0
    return H_local


def combine_global_local_incidence(
    h_global: torch.Tensor,
    h_local: torch.Tensor,
) -> torch.Tensor:
    """
    Combine global static H and per-window local dynamic H.

    h_global: (N, E_g) or (B, N, E_g)
    h_local:  (B, N, E_l)
    return:   (B, N, E_g + E_l)
    """
    if h_local.ndim != 3:
        raise ValueError(f"h_local must be 3D (B, N, E_l), got {tuple(h_local.shape)}")

    B, N, _ = h_local.shape
    if h_global.ndim == 2:
        if h_global.shape[0] != N:
            raise ValueError(
                f"h_global node dim mismatch: {h_global.shape[0]} vs local {N}"
            )
        h_global = h_global.unsqueeze(0).expand(B, -1, -1)
    elif h_global.ndim == 3:
        if h_global.shape[0] != B or h_global.shape[1] != N:
            raise ValueError(
                f"h_global shape {tuple(h_global.shape)} incompatible with "
                f"h_local shape {tuple(h_local.shape)}"
            )
    else:
        raise ValueError(
            f"h_global must be 2D or 3D, got shape={tuple(h_global.shape)}"
        )

    return torch.cat([h_global.float(), h_local.float()], dim=2)
