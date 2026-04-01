from pathlib import Path
import numpy as np


def load_smd_features(path: str | Path, delimiter: str = ",") -> np.ndarray:
    path = Path(path)
    X = np.loadtxt(path, delimiter=delimiter)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape={X.shape}")

    return X


def load_smd_labels(path: str | Path, delimiter: str = ",") -> np.ndarray:
    path = Path(path)
    y = np.loadtxt(path, delimiter=delimiter)

    if y.ndim != 1:
        y = np.asarray(y).reshape(-1)

    return y