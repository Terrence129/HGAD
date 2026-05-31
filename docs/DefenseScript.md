# English Defense Script for Cloud Monitoring Anomaly Detection Project

## Opening Statement (Core Project Overview)

My project is an unsupervised anomaly detection and diagnosis system for cloud computing monitoring data. I treat the 38 monitoring metrics of each server in the SMD dataset as nodes in a graph/hypergraph, detect anomalies by reconstructing normal behaviors, and compare four methods: Isolation Forest, LSTM-AE, GCN-AE, and the improved HGNN-AE.

## Project Structure

The core of my project is in the [HGAD](/Users/chenyaqi/Documents/FInal%20Project/HGAD) directory, with the following structure:

```text
HGAD/
├── data/              Data reading, normalization, sliding window processing, Dataset encapsulation
├── hypergraph/        Correlation matrix calculation, global/local hypergraph construction
├── model/             Model architectures (LSTM/GCN/HGNN, etc.)
├── evaluate/          Anomaly score calculation, threshold search, root-cause ranking
├── train_*.py         Independent training entry points for each model
├── scripts/           Batch experiment execution and visualization plotting
├── results/           results.csv, summary.csv, experiment logs, figures
└── docs/              Interim report, thesis draft, meeting notes
```

Key files to highlight:

- [data/preprocess.py](/Users/chenyaqi/Documents/FInal%20Project/HGAD/data/preprocess.py): Z-score standardization, sliding window segmentation, and window-level anomaly labeling.
- [hypergraph/build_hypergraph.py](/Users/chenyaqi/Documents/FInal%20Project/HGAD/hypergraph/build_hypergraph.py): Construct global hypergraph from training set correlations (also supports window-level local hypergraph).
- [model/hierarchical_hgnn.py](/Users/chenyaqi/Documents/FInal%20Project/HGAD/model/hierarchical_hgnn.py): Core model with AdaptiveHypergraphConv + autoencoder structure.
- [train_hier_hgnn.py](/Users/chenyaqi/Documents/FInal%20Project/HGAD/train_hier_hgnn.py): HGNN training, anomaly scoring, and root cause output.
- [scripts/run_batch_experiments.py](/Users/chenyaqi/Documents/FInal%20Project/HGAD/scripts/run_batch_experiments.py): Batch experiments on 28 servers with 4 models.
- [results/summary.csv](/Users/chenyaqi/Documents/FInal%20Project/HGAD/results/summary.csv): Final aggregated experimental results.

## Methodology Explanation

### Step 1: Data Processing

The SMD dataset provides training, test sets, and labels for each server. To avoid data leakage, I fit the `StandardScaler` **only on the training set** and apply it to transform the test set. Then I split the time series into fixed-length windows with `window=10, stride=1`; a window is labeled as anomalous if any timestamp within it is marked as abnormal.

### Step 2: Model Input

- For LSTM-AE: Input shape is `(window, features)`, directly using the time-series window as input.
- For GCN-AE/HGNN-AE: Each monitoring metric is treated as a node, and the node feature is the metric’s values across 10 time steps, resulting in an input shape of `(38 nodes, 10 time values)`.

### Step 3: Graph Structure

- GCN uses a pairwise correlation adjacency matrix, which only captures binary relationships between metrics.
- HGNN uses an incidence matrix `H`, where one hyperedge can connect multiple metrics—this is more suitable for cloud fault scenarios where "multiple monitoring metrics anomaly together". In batch experiments, I mainly used **global hypergraph** (constructed from training set correlations for stability); the code also supports global+local hybrid structures, but global hypergraph was adopted for large-scale experiments to ensure stability.

### Step 4: Anomaly Scoring

All models follow the autoencoder paradigm: learning to reconstruct normal behaviors on training data (only normal samples). During testing, high reconstruction error indicates anomalies. For HGNN-AE, I additionally incorporated temporal error and embedding deviation, with the final anomaly score being a weighted combination of these components. Instead of manually setting a fixed threshold, I searched for optimal thresholds from percentiles of training scores (e.g., 96, 97, 98, 99, 99.5 percentiles).

## Experimental Results

The average results across 28 servers are as follows:

```text
LSTM-AE:           Precision 0.426, Recall 0.564, F1 0.384
GCN-AE:            Precision 0.453, Recall 0.563, F1 0.380
Improved HGNN-AE:  Precision 0.375, Recall 0.639, F1 0.365
Isolation Forest:  Precision 0.376, Recall 0.328, F1 0.282
```

Key interpretation (avoid claiming "HGNN is universally the best"):
*LSTM-AE has the highest average F1 in the recorded batch, while Improved HGNN-AE achieves the highest recall and provides a clearer diagnostic path through metric-level reconstruction errors. Therefore, HGNN-AE is more suitable as an explainable second-stage detector rather than a direct replacement for all baselines.*

## Q&A Responses to Potential Questions

### Q1: Why use hypergraph instead of traditional graph?

A: Traditional graphs only capture pairwise relationships (e.g., metric A correlates with metric B). However, cloud system faults rarely stem from isolated changes of two metrics—instead, they often involve a group of metrics (e.g., CPU, memory, latency, and I/O anomalies simultaneously). A hyperedge in hypergraph can connect multiple metrics, making it more capable of expressing group-level dependencies. My model uses hypergraph convolution to propagate information across these metric groups, then detects anomalies via reconstruction error.

### Q2: Why is HGNN-AE’s F1 score lower than LSTM-AE?

A: This reflects a typical precision-recall trade-off. HGNN-AE achieves the highest recall, meaning it captures more true anomalies; however, its lower precision leads to more false positives, which drags down the F1 score. The reason may lie in the hypergraph structure being more sensitive to complex correlated changes, making it prone to triggering alerts. My conclusion is not that HGNN-AE completely outperforms LSTM-AE, but that it adds value in high recall and diagnostic interpretability—future work will focus on better threshold calibration and dynamic graph tuning to improve precision.

### Q3: What is the novelty of your work?

A: The novelty mainly lies in engineering implementation and combinatorial design:

1. Transforming multi-variate time series in SMD into a metric-node reconstruction task;
2. Using correlation-based hypergraph to model higher-order metric dependencies;
3. Integrating adaptive edge gates into the HGNN autoencoder, enabling the model to learn the importance of different hyperedges;
4. Beyond anomaly detection, providing root-cause style ranking based on metric-level reconstruction errors.

### Q4: Why are test scores extremely large in GCN logs (e.g., gcn__machine-1-1.log)?

A: Some servers contain unstable or near-constant metric dimensions, leading to numerical warnings during correlation computation—I handled this with `nan_to_num` to avoid crashes. The extremely large test reconstruction scores indicate significant distribution shifts or extreme anomalous windows in those servers. Since thresholds are selected from training scores, this results in high recall but also numerous predicted anomalies. This highlights threshold calibration as a critical direction for future improvement.

### Q5: Why choose window size = 10?

A: Window size 10 is a trade-off: a too-small window loses temporal patterns, while a too-large window increases training costs and coarsens anomaly localization. SMD is high-frequency monitoring data, so window=10 retains short-term trends while keeping the model lightweight—this is practical for batch experiments across 28 servers.

### Q6: Is there any data leakage in your pipeline?

A: No. The scaler is fit **only on training data** and then applied to the test set. The global graph/hypergraph structure is also constructed from training set correlations. Test labels are only used for evaluation and never involved in model training.

### Q7: Is your root cause detection truly identifying causal root causes?

A: Strictly speaking, it is not verified causal root cause. It is a root-cause style ranking based on metric-level reconstruction and temporal errors. It indicates which metrics contribute most to the anomaly score, thus improving interpretability. However, true causal validation would require labeled root-cause data or operator logs—this is one of my future research directions.

### Q8: How to deploy your model to real cloud systems?

A: I would not deploy HGNN-AE as the sole real-time detector initially. A practical deployment architecture would use a fast baseline detector for first-stage alerting, then leverage HGNN-AE as a second-stage diagnostic model for severe or ambiguous incidents. This balances efficiency, recall, and interpretability.

## Closing Remarks (Last 30 Seconds)

To conclude, this project implemented a complete anomaly detection pipeline—from SMD data preprocessing to model comparison and result visualization. LSTM-AE achieves the best average F1 score, while the improved HGNN-AE attains the highest recall and provides metric-level diagnostic insights. Therefore, the main contribution is not merely a higher performance score, but a more explainable hypergraph-based framework for detecting complex cloud faults.
