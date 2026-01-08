import numpy as np

def build_incidence_from_window(
    Xw: np.ndarray,  # (window, d)
    k: int = 5,
    tau: float = 0.7,
    eps: float = 1e-8
) -> np.ndarray:
    # 防止 std=0 导致 NaN：先做方差检查
    std = Xw.std(axis=0)
    safe = std > eps

    # 如果某些维度方差为0，先让它们不参与相关计算
    Xw_safe = Xw[:, safe]

    if Xw_safe.shape[1] < 2:
        # 极端情况：几乎都常数，返回一个空图
        d = Xw.shape[1]
        return np.zeros((d, 1), dtype=np.float32)

    C = np.corrcoef(Xw_safe.T)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    d_full = Xw.shape[1]
    d = Xw_safe.shape[1]

    edges = []
    for i in range(d):
        idx = np.argsort(-np.abs(C[i]))
        group = [i]
        for j in idx[1:k+1]:
            if np.abs(C[i, j]) >= tau:
                group.append(j)
        if len(group) > 1:
            edges.append(sorted(set(group)))

    # 去重
    uniq = []
    seen = set()
    for e in edges:
        t = tuple(e)
        if t not in seen:
            uniq.append(e)
            seen.add(t)

    # 映射回 full 维度
    cols = np.where(safe)[0]  # safe维度在原始d_full中的索引
    m = max(1, len(uniq))
    H = np.zeros((d_full, m), dtype=np.float32)

    for e_id, nodes in enumerate(uniq):
        orig_nodes = cols[nodes]
        H[orig_nodes, e_id] = 1.0

    return H