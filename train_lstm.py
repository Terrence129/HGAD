import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score

from data.load_smd import load_smd_features, load_smd_labels
from data.preprocess import zscore_normalize, apply_scaler, sliding_window, window_labels
from train.lstm_autoencoder_baseline import LSTMAutoencoder, compute_scores


def moving_average(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, np.ones(k) / k, mode="valid")


def parse_percentiles(x: str) -> list[float]:
    return [float(i.strip()) for i in x.split(",") if i.strip()]


def split_train_val(windows: np.ndarray, val_ratio: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    n = len(windows)
    n_val = max(int(n * val_ratio), 1)
    n_train = n - n_val
    return windows[:n_train], windows[n_train:]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    denoise_std: float = 0.0,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    criterion = nn.MSELoss()

    total_loss = 0.0
    total_count = 0

    for (batch_x,) in loader:
        batch_x = batch_x.to(device)  # (B, T, F)

        model_input = batch_x
        if train_mode and denoise_std > 0:
            model_input = batch_x + torch.randn_like(batch_x) * denoise_std

        batch_hat = model(model_input)
        loss = criterion(batch_hat, batch_x)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        bsz = batch_x.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz

    return total_loss / max(total_count, 1)


def train_model(
    model: nn.Module,
    w_train: np.ndarray,
    w_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    train_loader = DataLoader(
        TensorDataset(torch.tensor(w_train, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(w_val, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val = np.inf
    wait = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            denoise_std=args.denoise_std,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            denoise_std=0.0,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}, best val loss={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


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
    device = torch.device(args.device)
    percentiles = parse_percentiles(args.percentiles)

    train_path = f"data/raw/ServerMachineDataset/train/{args.machine}.txt"
    test_path = f"data/raw/ServerMachineDataset/test/{args.machine}.txt"
    label_path = f"data/raw/ServerMachineDataset/test_label/{args.machine}.txt"

    print("Loading data...")
    X_train_raw = load_smd_features(train_path)
    X_test_raw = load_smd_features(test_path)
    y_test_raw = load_smd_labels(label_path)

    print("Normalizing...")
    X_train, scaler = zscore_normalize(X_train_raw)
    X_test = apply_scaler(X_test_raw, scaler)

    print("Building sliding windows...")
    W_train = sliding_window(X_train, window=args.window, stride=args.stride)
    W_test = sliding_window(X_test, window=args.window, stride=args.stride)
    y_test = window_labels(y_test_raw, window=args.window, stride=args.stride)

    print(f"W_train_all shape: {W_train.shape}")
    print(f"W_test shape : {W_test.shape}")
    print(f"y_test shape : {y_test.shape}, positives={int(y_test.sum())}")

    W_train_split, W_val_split = split_train_val(W_train, val_ratio=args.val_ratio)
    print(f"W_train split: {W_train_split.shape}, W_val split: {W_val_split.shape}")

    model = LSTMAutoencoder(
        num_features=W_train.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print("\nTraining LSTM Autoencoder...")
    model = train_model(
        model=model,
        w_train=W_train_split,
        w_val=W_val_split,
        args=args,
        device=device,
    )

    print("\nComputing reconstruction scores...")
    train_scores = compute_scores(
        model=model,
        windows=W_train_split,
        batch_size=args.eval_batch_size,
        device=device,
    )
    test_scores = compute_scores(
        model=model,
        windows=W_test,
        batch_size=args.eval_batch_size,
        device=device,
    )

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
    parser = argparse.ArgumentParser(description="Train and evaluate LSTM AE baseline on SMD.")

    parser.add_argument("--machine", type=str, default="machine-1-1")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--denoise-std", type=float, default=0.0)
    parser.add_argument("--smooth-k", type=int, default=5)
    parser.add_argument("--percentiles", type=str, default="99.0,99.5,99.9")
    parser.add_argument("--device", type=str, default="cpu")

    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
