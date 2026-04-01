import numpy as np
from torch.utils.data import DataLoader

from data.load_smd import load_smd_features, load_smd_labels
from data.preprocess import (
    zscore_normalize,
    apply_scaler,
    sliding_window,
    window_labels,
)
from data.dataset import SMDTrainDataset, SMDTestDataset
from hypergraph.build_hypergraph import (
    compute_correlation_matrix,
    build_incidence_from_correlation,
    window_to_node_features,
)

import torch
from model.hgnn import HGNN_AE

def main():
    print("HGAD data pipeline check")

    machine = "machine-1-1"
    window = 10
    stride = 1
    batch_size = 64

    train_path = f"data/raw/ServerMachineDataset/train/{machine}.txt"
    test_path = f"data/raw/ServerMachineDataset/test/{machine}.txt"
    label_path = f"data/raw/ServerMachineDataset/test_label/{machine}.txt"

    # 1. load raw data
    X_train_raw = load_smd_features(train_path)
    X_test_raw = load_smd_features(test_path)
    y_test_raw = load_smd_labels(label_path)

    print("train raw:", X_train_raw.shape)
    print("test raw :", X_test_raw.shape)
    print("label raw:", y_test_raw.shape)

    # 2. normalize: fit on train only
    X_train, scaler = zscore_normalize(X_train_raw)
    X_test = apply_scaler(X_test_raw, scaler)

    # 3. sliding windows
    W_train = sliding_window(X_train, window=window, stride=stride)
    W_test = sliding_window(X_test, window=window, stride=stride)

    # 4. window labels
    y_test = window_labels(y_test_raw, window=window, stride=stride)

    print("W_train:", W_train.shape)
    print("W_test :", W_test.shape)
    print("y_test :", y_test.shape)

    print("train np.nan:", np.isnan(W_train).any())
    print("test np.nan :", np.isnan(W_test).any())

    assert W_test.shape[0] == y_test.shape[0], \
        f"Mismatch: W_test={W_test.shape[0]}, y_test={y_test.shape[0]}"

    print("Positive test windows:", y_test.sum())

    # 5. build datasets
    train_dataset = SMDTrainDataset(W_train)
    test_dataset = SMDTestDataset(W_test, y_test)

    print("train dataset size:", len(train_dataset))
    print("test dataset size :", len(test_dataset))

    # 6. build dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 7. inspect one batch
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    print("train batch shape:", train_batch.shape)

    x_test_batch, y_test_batch = test_batch
    print("test batch x shape:", x_test_batch.shape)
    print("test batch y shape:", y_test_batch.shape)
    print("test batch positive labels:", y_test_batch.sum().item())

    print("DataLoader OK")

    # 8. build hypergraph from normalized train data
    corr = compute_correlation_matrix(X_train)
    H = build_incidence_from_correlation(corr, threshold=0.8, top_k=6)

    print("corr shape:", corr.shape)
    print("H shape   :", H.shape)
    print("H density :", H.mean())

    # 9. inspect one sample -> node feature matrix
    sample_window = W_train[0]   # shape: (10, 38)
    X_nodes = window_to_node_features(sample_window)  # shape: (38, 10)

    print("sample window shape:", sample_window.shape)
    print("node feature shape :", X_nodes.shape)

    # 10. HGNN forward test
    model = HGNN_AE(in_dim=10)

    X_nodes_tensor = torch.tensor(X_nodes)
    H_tensor = torch.tensor(H)

    X_hat, Z = model(X_nodes_tensor, H_tensor)

    print("input shape :", X_nodes_tensor.shape)
    print("recon shape :", X_hat.shape)
    print("embedding   :", Z.shape)

    loss_fn = torch.nn.MSELoss()

    loss = loss_fn(X_hat, X_nodes_tensor)

    print("reconstruction loss:", loss.item())

if __name__ == "__main__":
    main()