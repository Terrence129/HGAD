from pathlib import Path
import numpy as np

def load_smd_txt(path: str | Path, delimiter: str = ",") -> np.ndarray:
    path = Path(path)
    X = np.loadtxt(path, delimiter=delimiter)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={X.shape}")
    return X