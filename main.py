def main():
    print("HGAD project bootstrap OK")

if __name__ == "__main__":
    main()

from data.load_smd import load_smd_txt

X = load_smd_txt("data/raw/ServerMachineDataset/train/machine-1-1.txt")
print("Loaded:", X.shape)

from data.preprocess import zscore_normalize, sliding_window
Xn, _ = zscore_normalize(X)
W = sliding_window(Xn, window=10, stride=1)
print("Windows:", W.shape)

from hypergraph.build_hypergraph import build_incidence_from_window
H = build_incidence_from_window(W[0], k=5, tau=0.7)
print("Incidence H:", H.shape, "density:", H.mean())

from train.train_detector import train_detector
model = train_detector(W[:200], epochs=3)  # 先用少量样本跑通

import matplotlib.pyplot as plt
from evaluate.anomaly_score import compute_scores
scores = compute_scores(model, W[:500])
plt.plot(scores)
plt.title("Anomaly score (reconstruction MSE)")
plt.show()