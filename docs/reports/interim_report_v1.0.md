# Interim Report

- Report Version: v1.0
- Date: 2026-05-05
- Author: [Your Name]

## 1. Project Overview

The HGAD project develops a Heterogeneous Graph based Anomaly Detection framework for cloud systems. The implementation uses Python with PyTorch and PyTorch Geometric to model server machine behavior from the SMD dataset and detect anomalies in cloud performance metrics.

## 2. Completed Work

- Established project repository and version control on GitHub.
- Implemented the data loading pipeline for the SMD dataset.
- Developed preprocessing utilities for z-score normalization, sliding window generation, and label aggregation.
- Created initial PyTorch dataset wrappers and data inspection routines.

## 3. Ongoing Work

- Building the HGAD model architecture with heterogeneous graph representation.
- Integrating PyG-based graph operations for anomaly scoring.
- Defining evaluation metrics and baseline comparisons for experiments.

## 4. Planned Work

- Conduct experiments on selected SMD machine instances.
- Perform ablation studies on graph construction and window settings.
- Draft thesis chapters for methodology, experiments, and discussion.

## 5. Issues and Challenges

- Ensuring proper data alignment between time windows and anomaly labels.
- Selecting robust graph construction methods for heterogeneous relationships.
- Maintaining reproducibility and backup of intermediate results.

## Notes

> [Placeholder: Add any additional remarks or supervisor comments here.]
