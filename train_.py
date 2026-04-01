import torch
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader

from data.dataset import SMDTrainDataset
from data.load_smd import load_smd_features, load_smd_labels
from data.preprocess import (
    zscore_normalize,
    apply_scaler,
    sliding_window,
    window_labels,
)

from hypergraph.build_hypergraph import (
    compute_correlation_matrix,
    build_incidence_from_correlation,
    window_to_node_features,
)

from model.hgnn import HGNN_AE

from sklearn.metrics import precision_score, recall_score, f1_score


device = torch.device("cpu")


# ================================
# Score smoothing
# ================================
def moving_average(x, k=5):
    pad = k // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, np.ones(k) / k, mode="valid")


# ================================
# Compute reconstruction scores
# ================================
def compute_window_scores(model, windows, H):
    scores = []

    model.eval()

    with torch.no_grad():
        for window_x in windows:
            X_nodes = window_to_node_features(window_x)
            X_nodes = torch.tensor(X_nodes, dtype=torch.float32).to(device)

            X_hat, _ = model(X_nodes, H)

            node_error = torch.mean((X_nodes - X_hat) ** 2, dim=1)
            window_score = node_error.mean().item()

            scores.append(window_score)

    return np.array(scores)


# ================================
# Root cause for a single window
# ================================
def get_root_cause_for_window(model, window_x, H, top_k=3):
    model.eval()

    with torch.no_grad():
        X_nodes = window_to_node_features(window_x)
        X_nodes = torch.tensor(X_nodes, dtype=torch.float32).to(device)

        X_hat, _ = model(X_nodes, H)

        node_error = torch.mean((X_nodes - X_hat) ** 2, dim=1)
        node_error = node_error.detach().cpu().numpy()

        top_idx = np.argsort(node_error)[::-1][:top_k]

    return top_idx, node_error


# ================================
# Detailed root cause report
# ================================
def detect_root_causes(model, windows, pred, H, metric_names, top_k=3, max_windows=10):
    print("\nRoot Cause Detection")
    print("-" * 40)

    anomaly_indices = np.where(pred == 1)[0]

    if len(anomaly_indices) == 0:
        print("No anomaly windows detected.")
        return []

    selected_indices = anomaly_indices[:max_windows]
    all_top_metrics = []

    for rank, win_idx in enumerate(selected_indices, start=1):
        top_idx, node_error = get_root_cause_for_window(model, windows[win_idx], H, top_k=top_k)

        print(f"\nAnomaly window #{rank} (test index={win_idx})")
        for idx in top_idx:
            metric_name = metric_names[idx]
            error_value = node_error[idx]
            print(f"{metric_name:<12} error={error_value:.6f}")
            all_top_metrics.append(metric_name)

    return all_top_metrics


# ================================
# Global root cause ranking
# ================================
def global_root_cause_ranking(model, windows, pred, H, metric_names, top_k=3):
    anomaly_indices = np.where(pred == 1)[0]

    if len(anomaly_indices) == 0:
        print("\nGlobal Root Cause Ranking")
        print("-" * 40)
        print("No anomaly windows detected.")
        return

    counter = Counter()

    model.eval()

    with torch.no_grad():
        for win_idx in anomaly_indices:
            top_idx, _ = get_root_cause_for_window(model, windows[win_idx], H, top_k=top_k)
            for idx in top_idx:
                counter[metric_names[idx]] += 1

    print("\nGlobal Root Cause Ranking")
    print("-" * 40)

    for metric_name, freq in counter.most_common(10):
        print(f"{metric_name:<12} count={freq}")


# ================================
# Main
# ================================
def main():
    machine = "machine-1-1"
    window = 10
    batch_size = 64
    epochs = 10

    # 临时指标名，后面你可以替换成真实监控指标名
    metric_names = [f"metric_{i}" for i in range(38)]

    train_path = f"data/raw/ServerMachineDataset/train/{machine}.txt"

    # ----------------------------
    # Load training data
    # ----------------------------
    X_train_raw = load_smd_features(train_path)

    # normalize
    X_train, scaler = zscore_normalize(X_train_raw)

    # sliding windows
    W_train = sliding_window(X_train, window)

    dataset = SMDTrainDataset(W_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ----------------------------
    # Build hypergraph
    # ----------------------------
    corr = compute_correlation_matrix(X_train)
    H = build_incidence_from_correlation(corr, threshold=0.8, top_k=6)
    H = torch.tensor(H, dtype=torch.float32).to(device)

    # ----------------------------
    # Model
    # ----------------------------
    model = HGNN_AE(in_dim=10).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # =============================
    # TRAIN
    # =============================
    for epoch in range(epochs):
        total_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()

            loss_batch = 0.0

            for window_x in batch:
                X_nodes = window_to_node_features(window_x.numpy())
                X_nodes = torch.tensor(X_nodes, dtype=torch.float32).to(device)

                X_hat, _ = model(X_nodes, H)

                loss = loss_fn(X_hat, X_nodes)
                loss_batch += loss

            loss_batch = loss_batch / len(batch)

            loss_batch.backward()
            optimizer.step()

            total_loss += loss_batch.item()

        print(f"Epoch {epoch+1} | loss={total_loss:.4f}")

    # =============================
    # EVALUATION
    # =============================

    # train scores -> threshold
    train_scores = compute_window_scores(model, W_train, H)

    print("\nTrain score stats")
    print("mean:", train_scores.mean())
    print("max :", train_scores.max())

    # ----------------------------
    # Load test data
    # ----------------------------
    test_path = f"data/raw/ServerMachineDataset/test/{machine}.txt"
    label_path = f"data/raw/ServerMachineDataset/test_label/{machine}.txt"

    X_test_raw = load_smd_features(test_path)
    y_test_raw = load_smd_labels(label_path)

    X_test = apply_scaler(X_test_raw, scaler)

    W_test = sliding_window(X_test, window)
    y_test = window_labels(y_test_raw, window)

    # ----------------------------
    # Compute test scores
    # ----------------------------
    test_scores = compute_window_scores(model, W_test, H)

    print("\nTest score stats")
    print("mean:", test_scores.mean())
    print("max :", test_scores.max())

    # ----------------------------
    # Score smoothing
    # ----------------------------
    test_scores_smooth = moving_average(test_scores, k=5)

    print("\nSmoothed score stats")
    print("mean:", test_scores_smooth.mean())
    print("max :", test_scores_smooth.max())

    # ----------------------------
    # Try multiple thresholds
    # ----------------------------
    print("\nEvaluation with different thresholds\n")

    best_f1 = -1.0
    best_q = None
    best_threshold = None
    best_pred = None
    best_precision = None
    best_recall = None

    #讲清楚阈值判定的原因，为什么选择百分位数，以及为什么选择这些百分位数
    for q in [99.0, 99.5, 99.9]:
        threshold = np.percentile(train_scores, q)
        pred = (test_scores_smooth > threshold).astype(int)

        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)

        print(f"Percentile {q}")
        print("Threshold:", threshold)
        print("Predicted anomalies:", pred.sum())
        print("True anomalies     :", y_test.sum())
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1 score :", f1)
        print("-" * 40)

        if f1 > best_f1:
            best_f1 = f1
            best_q = q
            best_threshold = threshold
            best_pred = pred
            best_precision = precision
            best_recall = recall

    print("\nBest threshold setting")
    print("-" * 40)
    print("Best percentile:", best_q)
    print("Best threshold :", best_threshold)
    print("Best precision :", best_precision)
    print("Best recall    :", best_recall)
    print("Best F1 score  :", best_f1)

    # =============================
    # ROOT CAUSE DETECTION ！！！！！！！！！！
    # =============================
    detect_root_causes(
        model=model,
        windows=W_test,
        pred=best_pred,
        H=H,
        metric_names=metric_names,
        top_k=3,
        max_windows=10,
    )

    global_root_cause_ranking(
        model=model,
        windows=W_test,
        pred=best_pred,
        H=H,
        metric_names=metric_names,
        top_k=3,
    )


if __name__ == "__main__":
    main()