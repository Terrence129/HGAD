import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.load_smd import load_smd_features, load_smd_labels
from data.preprocess import zscore_normalize, apply_scaler, sliding_window, window_labels
from hypergraph.build_hypergraph import (
    compute_correlation_matrix,
    build_incidence_from_correlation,
    build_local_incidence_from_node_features,
)
from model.hierarchical_hgnn import HierarchicalAdaptiveHGNN_AE
from evaluate.hierarchical_eval import (
    build_hybrid_incidence,
    moving_average,
    compute_embedding_stats,
    compute_anomaly_scores,
    search_threshold_by_percentile,
    topk_root_causes_for_window,
    global_root_cause_ranking,
)


def parse_percentiles(x: str) -> list[float]:
    return [float(i.strip()) for i in x.split(",") if i.strip()]


def split_train_val(windows: np.ndarray, val_ratio: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    n = len(windows)
    n_val = max(int(n * val_ratio), 1)
    n_train = n - n_val
    return windows[:n_train], windows[n_train:]


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    h_global: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    graph_mode: str = "global",
    local_top_k: int = 4,
    temporal_loss_weight: float = 0.2,
    latent_reg_weight: float = 1e-3,
    denoise_std: float = 0.03,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_recon = 0.0
    total_temp = 0.0
    total_latent = 0.0
    total_count = 0

    for (batch_w,) in loader:
        batch_w = batch_w.to(device)  # (B, T, F)
        x_nodes = batch_w.transpose(1, 2)  # (B, F, T)

        h_batch = build_hybrid_incidence(
            x_nodes=x_nodes,
            h_global_t=h_global,
            graph_mode=graph_mode,
            local_top_k=local_top_k,
        )

        model_input = x_nodes
        if train_mode and denoise_std > 0:
            model_input = x_nodes + torch.randn_like(x_nodes) * denoise_std

        x_hat, z, _ = model(model_input, h_batch)

        loss_recon = torch.mean((x_hat - x_nodes) ** 2)
        loss_temp = torch.mean((torch.diff(x_hat, dim=2) - torch.diff(x_nodes, dim=2)) ** 2)
        loss_latent = torch.mean(z ** 2)
        loss = loss_recon + temporal_loss_weight * loss_temp + latent_reg_weight * loss_latent

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        bsz = batch_w.size(0)
        total_loss += loss.item() * bsz
        total_recon += loss_recon.item() * bsz
        total_temp += loss_temp.item() * bsz
        total_latent += loss_latent.item() * bsz
        total_count += bsz

    return {
        "loss": total_loss / max(total_count, 1),
        "recon": total_recon / max(total_count, 1),
        "temp": total_temp / max(total_count, 1),
        "latent": total_latent / max(total_count, 1),
    }


def train_model(
    model: torch.nn.Module,
    w_train: np.ndarray,
    w_val: np.ndarray,
    h_global: np.ndarray,
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

    h_global_t = torch.tensor(h_global, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = np.inf
    wait = 0

    for epoch in range(1, args.epochs + 1):
        train_info = run_epoch(
            model=model,
            loader=train_loader,
            h_global=h_global_t,
            optimizer=optimizer,
            device=device,
            graph_mode=args.graph_mode,
            local_top_k=args.local_top_k,
            temporal_loss_weight=args.temporal_loss_weight,
            latent_reg_weight=args.latent_reg_weight,
            denoise_std=args.denoise_std,
        )
        val_info = run_epoch(
            model=model,
            loader=val_loader,
            h_global=h_global_t,
            optimizer=None,
            device=device,
            graph_mode=args.graph_mode,
            local_top_k=args.local_top_k,
            temporal_loss_weight=args.temporal_loss_weight,
            latent_reg_weight=args.latent_reg_weight,
            denoise_std=0.0,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_info['loss']:.6f} "
            f"(recon={train_info['recon']:.6f}, temp={train_info['temp']:.6f}, latent={train_info['latent']:.6f}) | "
            f"val_loss={val_info['loss']:.6f} "
            f"(recon={val_info['recon']:.6f}, temp={val_info['temp']:.6f}, latent={val_info['latent']:.6f})"
        )

        if val_info["loss"] < best_val:
            best_val = val_info["loss"]
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


def main():
    parser = argparse.ArgumentParser("Train enhanced hierarchical adaptive HGNN for SMD.")
    parser.add_argument("--machine", type=str, default="machine-1-1")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--latent-noise-std", type=float, default=0.05)
    parser.add_argument("--latent-reg-weight", type=float, default=1e-3)
    parser.add_argument("--denoise-std", type=float, default=0.03)
    parser.add_argument("--graph-mode", type=str, default="global", choices=["global", "global_local"])
    parser.add_argument("--local-top-k", type=int, default=4)
    parser.add_argument("--global-threshold", type=float, default=0.8)
    parser.add_argument("--global-top-k", type=int, default=6)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--temporal-loss-weight", type=float, default=0.2)
    parser.add_argument("--score-weights", type=str, default="0.75,0.2,0.05")
    parser.add_argument("--percentiles", type=str, default="99.0,99.5,99.9")
    parser.add_argument("--smooth-k", type=int, default=5)
    parser.add_argument("--top-k-root", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    score_weights = tuple(float(x.strip()) for x in args.score_weights.split(","))
    percentiles = parse_percentiles(args.percentiles)

    metric_names = [f"metric_{i}" for i in range(38)]

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
    h_global = build_incidence_from_correlation(
        corr,
        threshold=args.global_threshold,
        top_k=args.global_top_k,
    ).astype(np.float32)
    print(f"Global H shape: {h_global.shape}")

    model = HierarchicalAdaptiveHGNN_AE(
        in_dim=args.window,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        latent_noise_std=args.latent_noise_std,
        use_input_skip=False,
    ).to(device)

    model = train_model(model, w_train, w_val, h_global, args, device)

    emb_stats = compute_embedding_stats(
        model=model,
        windows=w_train,
        h_global=h_global,
        batch_size=args.batch_size,
        device=device,
        local_top_k=args.local_top_k,
        graph_mode=args.graph_mode,
    )

    train_scores_out = compute_anomaly_scores(
        model=model,
        windows=w_train,
        h_global=h_global,
        emb_stats=emb_stats,
        score_weights=score_weights,
        batch_size=args.batch_size,
        device=device,
        local_top_k=args.local_top_k,
        graph_mode=args.graph_mode,
    )
    test_scores_out = compute_anomaly_scores(
        model=model,
        windows=w_test,
        h_global=h_global,
        emb_stats=emb_stats,
        score_weights=score_weights,
        batch_size=args.batch_size,
        device=device,
        local_top_k=args.local_top_k,
        graph_mode=args.graph_mode,
    )

    train_total = train_scores_out["window_total"]
    test_total = test_scores_out["window_total"]
    test_total_smooth = moving_average(test_total, k=args.smooth_k)

    best = search_threshold_by_percentile(
        train_scores=train_total,
        test_scores=test_total_smooth,
        y_test=y_test,
        percentiles=percentiles,
    )

    print("\nBest threshold setting")
    print("-" * 40)
    print(f"Percentile : {best['percentile']}")
    print(f"Threshold  : {best['threshold']:.6f}")
    print(f"Precision  : {best['precision']:.6f}")
    print(f"Recall     : {best['recall']:.6f}")
    print(f"F1-score   : {best['f1']:.6f}")
    print(f"Pred count : {int(best['pred'].sum())}")
    print(f"True count : {int(y_test.sum())}")

    pred = best["pred"]
    anomaly_idx = np.where(pred == 1)[0]

    print("\nRoot Cause Detection")
    print("-" * 40)
    if len(anomaly_idx) == 0:
        print("No anomaly windows detected.")
    else:
        first_idx = int(anomaly_idx[0])
        x_nodes = w_test[first_idx].T.astype(np.float32)
        if args.graph_mode == "global_local":
            h_local = build_local_incidence_from_node_features(x_nodes, top_k=args.local_top_k)
            h_hybrid = np.concatenate([h_global, h_local], axis=1)
        else:
            h_hybrid = h_global

        topk = topk_root_causes_for_window(
            node_recon=test_scores_out["node_recon"][first_idx],
            node_temporal=test_scores_out["node_temporal"][first_idx],
            h_incidence=h_hybrid,
            metric_names=metric_names,
            top_k=args.top_k_root,
        )
        print(f"Single-window top-{args.top_k_root} root causes (window idx={first_idx})")
        for name, score in topk:
            print(f"{name:<12} score={score:.6f}")

        ranking = global_root_cause_ranking(
            all_node_recon=test_scores_out["node_recon"],
            all_node_temporal=test_scores_out["node_temporal"],
            pred=pred,
            h_global=h_global,
            metric_names=metric_names,
            top_k_each=args.top_k_root,
        )
        print("\nGlobal root cause ranking (top 10)")
        for name, cnt in ranking[:10]:
            print(f"{name:<12} count={cnt}")


if __name__ == "__main__":
    main()
