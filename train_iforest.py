import argparse
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

from data.load_smd import load_smd_features, load_smd_labels
from data.preprocess import zscore_normalize, apply_scaler, sliding_window, window_labels


def moving_average(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, np.ones(k) / k, mode="valid")


def parse_percentiles(x: str) -> list[float]:
    return [float(i.strip()) for i in x.split(",") if i.strip()]


def window_to_flat_features(windows: np.ndarray) -> np.ndarray:
    if windows.ndim != 3:
        raise ValueError(f"Expected windows shape (N, T, F), got {windows.shape}")
    n = windows.shape[0]
    return windows.reshape(n, -1).astype(np.float32)


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

        print("-" * 40)
        print(f"Percentile: {q}")
        print(f"Threshold : {threshold:.6f}")
        print(f"Pred count: {int(pred.sum())}")
        print(f"True count: {int(y_test.sum())}")
        print(f"Precision : {precision:.6f}")
        print(f"Recall    : {recall:.6f}")
        print(f"F1        : {f1:.6f}")

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


def run(args: argparse.Namespace) -> None:
    percentiles = parse_percentiles(args.percentiles)

    train_path = f"data/raw/ServerMachineDataset/train/{args.machine}.txt"
    test_path = f"data/raw/ServerMachineDataset/test/{args.machine}.txt"
    label_path = f"data/raw/ServerMachineDataset/test_label/{args.machine}.txt"

    print("Loading data...")
    x_train_raw = load_smd_features(train_path)
    x_test_raw = load_smd_features(test_path)
    y_test_raw = load_smd_labels(label_path)

    print("Normalizing...")
    x_train, scaler = zscore_normalize(x_train_raw)
    x_test = apply_scaler(x_test_raw, scaler)

    print("Building sliding windows...")
    w_train = sliding_window(x_train, window=args.window, stride=args.stride)
    w_test = sliding_window(x_test, window=args.window, stride=args.stride)
    y_test = window_labels(y_test_raw, window=args.window, stride=args.stride)

    print(f"W_train shape: {w_train.shape}")
    print(f"W_test shape : {w_test.shape}")
    print(f"y_test shape : {y_test.shape}, positives={int(y_test.sum())}")

    x_train_flat = window_to_flat_features(w_train)  # (N_train, 380)
    x_test_flat = window_to_flat_features(w_test)    # (N_test, 380)
    print(f"X_train_flat : {x_train_flat.shape}")
    print(f"X_test_flat  : {x_test_flat.shape}")

    print("\nTraining Isolation Forest...")
    model = IsolationForest(
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        contamination=args.contamination,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    model.fit(x_train_flat)

    print("\nComputing anomaly scores...")
    # score_samples: larger -> more normal; convert to anomaly score by negating.
    train_scores = -model.score_samples(x_train_flat)
    test_scores = -model.score_samples(x_test_flat)

    test_scores_eval = moving_average(test_scores, k=args.smooth_k)
    print(f"Applied moving average smoothing: k={args.smooth_k}")

    print("\nScore stats")
    print(f"Train mean={train_scores.mean():.6f} max={train_scores.max():.6f}")
    print(f"Test  mean={test_scores_eval.mean():.6f} max={test_scores_eval.max():.6f}")

    print("\nEvaluation with thresholds")
    best = search_threshold_by_percentile(
        train_scores=train_scores,
        test_scores=test_scores_eval,
        y_test=y_test,
        percentiles=percentiles,
    )

    print("\nBest threshold setting")
    print("-" * 40)
    print(f"Best percentile: {best['percentile']}")
    print(f"Best threshold : {best['threshold']:.6f}")
    print(f"Best precision : {best['precision']:.6f}")
    print(f"Best recall    : {best['recall']:.6f}")
    print(f"Best F1        : {best['f1']:.6f}")
    print(f"Pred anomalies : {int(best['pred'].sum())}")
    print(f"True anomalies : {int(y_test.sum())}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate Isolation Forest baseline on SMD.")

    parser.add_argument("--machine", type=str, default="machine-1-1")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-samples", type=str, default="auto")
    parser.add_argument("--contamination", type=str, default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--smooth-k", type=int, default=5)
    parser.add_argument("--percentiles", type=str, default="99.0,99.5,99.9")

    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
