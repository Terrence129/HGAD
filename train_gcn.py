import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score

from data.load_smd import load_smd_features, load_smd_labels
from data.preprocess import zscore_normalize, apply_scaler, sliding_window, window_labels
from hypergraph.build_hypergraph import compute_correlation_matrix
from model.gcn_ae import GCNAutoencoder


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


def build_adjacency_from_correlation(
    corr: np.ndarray,
    threshold: float = 0.8,
    top_k: int = 6,
) -> np.ndarray:
    """
    Build binary adjacency A from feature correlation matrix.
    corr: (N, N)
    return A: (N, N)
    """
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"corr must be square, got shape={corr.shape}")

    n = corr.shape[0]
    a = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        row = np.abs(corr[i]).copy()
        row[i] = -1.0

        cand = np.where(row >= threshold)[0]
        if len(cand) > 0:
            cand = cand[np.argsort(row[cand])[::-1]]
            cand = cand[:top_k]
            a[i, cand] = 1.0

    # undirected + self-loop
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 1.0)
    return a


def normalize_adjacency(a: np.ndarray) -> np.ndarray:
    """
    A_hat = D^{-1/2} A D^{-1/2}
    """
    deg = np.sum(a, axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(deg + 1e-6)
    d_inv_sqrt = np.diag(d_inv_sqrt.astype(np.float32))
    return d_inv_sqrt @ a @ d_inv_sqrt


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    a_hat: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    denoise_std: float = 0.0,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_count = 0

    for (batch_w,) in loader:
        batch_w = batch_w.to(device)  # (B, T, F)
        x_nodes = batch_w.transpose(1, 2)  # (B, N, T)

        model_input = x_nodes
        if train_mode and denoise_std > 0:
            model_input = x_nodes + torch.randn_like(x_nodes) * denoise_std

        x_hat, _ = model(model_input, a_hat)
        loss = torch.mean((x_hat - x_nodes) ** 2)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        bsz = batch_w.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz

    return total_loss / max(total_count, 1)


def train_model(
    model: torch.nn.Module,
    w_train: np.ndarray,
    w_val: np.ndarray,
    a_hat: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
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

    a_hat_t = torch.tensor(a_hat, dtype=torch.float32, device=device)
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
            a_hat=a_hat_t,
            optimizer=optimizer,
            device=device,
            denoise_std=args.denoise_std,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            a_hat=a_hat_t,
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


@torch.no_grad()
def compute_scores(
    model: torch.nn.Module,
    windows: np.ndarray,
    a_hat: np.ndarray,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    model.eval()
    device = torch.device(device)

    w_t = torch.tensor(windows, dtype=torch.float32)
    a_hat_t = torch.tensor(a_hat, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(w_t), batch_size=batch_size, shuffle=False)

    scores = []
    for (batch_w,) in loader:
        batch_w = batch_w.to(device)  # (B, T, F)
        x_nodes = batch_w.transpose(1, 2)  # (B, N, T)
        x_hat, _ = model(x_nodes, a_hat_t)
        batch_scores = torch.mean((x_hat - x_nodes) ** 2, dim=(1, 2))
        scores.append(batch_scores.cpu())

    return torch.cat(scores, dim=0).numpy()


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


def main():
    parser = argparse.ArgumentParser("Train and evaluate GCN Autoencoder baseline on SMD.")
    parser.add_argument("--machine", type=str, default="machine-1-1")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--denoise-std", type=float, default=0.0)
    parser.add_argument("--corr-threshold", type=float, default=0.8)
    parser.add_argument("--corr-top-k", type=int, default=6)
    parser.add_argument("--smooth-k", type=int, default=5)
    parser.add_argument("--percentiles", type=str, default="99.0,99.5,99.9")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    percentiles = parse_percentiles(args.percentiles)

    train_path = f"data/raw/ServerMachineDataset/train/{args.machine}.txt"
    test_path = f"data/raw/ServerMachineDataset/test/{args.machine}.txt"
    label_path = f"data/raw/ServerMachineDataset/test_label/{args.machine}.txt"

    print("Loading SMD data...")
    x_train_raw = load_smd_features(train_path)
    x_test_raw = load_smd_features(test_path)
    y_test_raw = load_smd_labels(label_path)

    x_train, scaler = zscore_normalize(x_train_raw)
    x_test = apply_scaler(x_test_raw, scaler)

    w_train_all = sliding_window(x_train, window=args.window, stride=args.stride)
    w_test = sliding_window(x_test, window=args.window, stride=args.stride)
    y_test = window_labels(y_test_raw, window=args.window, stride=args.stride)

    print(f"W_train_all: {w_train_all.shape}")
    print(f"W_test     : {w_test.shape}")
    print(f"y_test     : {y_test.shape}, positives={int(y_test.sum())}")

    w_train, w_val = split_train_val(w_train_all, val_ratio=args.val_ratio)
    print(f"W_train split: {w_train.shape}, W_val split: {w_val.shape}")

    corr = compute_correlation_matrix(x_train)
    a = build_adjacency_from_correlation(
        corr,
        threshold=args.corr_threshold,
        top_k=args.corr_top_k,
    )
    a_hat = normalize_adjacency(a)
    print(f"A shape: {a.shape}, density={a.mean():.4f}")

    model = GCNAutoencoder(
        in_dim=args.window,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
    ).to(device)

    model = train_model(
        model=model,
        w_train=w_train,
        w_val=w_val,
        a_hat=a_hat,
        args=args,
        device=device,
    )

    train_scores = compute_scores(
        model=model,
        windows=w_train,
        a_hat=a_hat,
        batch_size=args.eval_batch_size,
        device=device,
    )
    test_scores = compute_scores(
        model=model,
        windows=w_test,
        a_hat=a_hat,
        batch_size=args.eval_batch_size,
        device=device,
    )

    test_scores_smooth = moving_average(test_scores, k=args.smooth_k)
    print(f"Applied moving average smoothing: k={args.smooth_k}")

    print("\nScore stats")
    print(f"Train mean={train_scores.mean():.6f} max={train_scores.max():.6f}")
    print(f"Test  mean={test_scores_smooth.mean():.6f} max={test_scores_smooth.max():.6f}")

    print("\nEvaluation with thresholds")
    best = search_threshold_by_percentile(
        train_scores=train_scores,
        test_scores=test_scores_smooth,
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


if __name__ == "__main__":
    main()
