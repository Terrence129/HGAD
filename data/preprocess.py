import numpy as np
from sklearn.preprocessing import StandardScaler

def zscore_normalize(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    return Xn, scaler

def sliding_window(X: np.ndarray, window: int = 10, stride: int = 1) -> np.ndarray:
    T = X.shape[0]
    if T <= window:
        raise ValueError("Time length must be > window")
    windows = []
    for i in range(0, T - window + 1, stride):
        windows.append(X[i:i+window])
    return np.stack(windows, axis=0)  # (N, window, d)