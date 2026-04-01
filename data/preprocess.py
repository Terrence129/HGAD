import numpy as np
from sklearn.preprocessing import StandardScaler


def zscore_normalize(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    return Xn, scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X)


def sliding_window(X: np.ndarray, window: int = 10, stride: int = 1) -> np.ndarray:
    T = X.shape[0]
    if T < window:
        raise ValueError(f"Time length must be >= window, got T={T}, window={window}")

    windows = []
    for i in range(0, T - window + 1, stride):
        windows.append(X[i:i + window])
    return np.stack(windows, axis=0)  # (N, window, d)

# 讲一下窗口数量选择的原因
def window_labels(labels: np.ndarray, window: int = 10, stride: int = 1) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)  # 保证是一维
    T = labels.shape[0]

    if T < window:
        raise ValueError(f"Label length must be >= window, got T={T}, window={window}")

    y = []
    for i in range(0, T - window + 1, stride):
        seg = labels[i:i + window]
        y.append(1 if np.max(seg) > 0 else 0)

    return np.array(y, dtype=np.int64)