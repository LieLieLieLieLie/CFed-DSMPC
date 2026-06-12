# CFed-DSMPC

Code release for **Clustered Federated Distribution-Shift-Aware Stochastic Model Predictive Control (CFed-DSMPC)**.

The project implements a heterogeneous multi-agent navigation benchmark and compares the proposed CFed-DSMPC controller against several MPC and learning-based baselines.

## Overview

CFed-DSMPC combines:

- clustered federated learning for heterogeneous agent groups;
- a density-estimation model for distribution-shift awareness;
- model predictive control with adaptive safety penalties;
- a structured obstacle environment with width-dependent feasible corridors.

The code is organized as a compact research prototype. Running the main script executes the experiments, stores numerical results, and generates plots used for analysis.

## Files

| File | Description |
| --- | --- |
| `config.py` | Global configuration, random seeds, simulation constants, agent types, and output paths. |
| `models.py` | Neural density estimator used for distribution-shift scoring. |
| `utils.py` | Vehicle dynamics, obstacle environment, geometry utilities, and plotting helpers. |
| `controllers.py` | Baseline MPC controllers and the proposed shift-aware controller. |
| `federated.py` | Federated clients, clustered FedAvg server, and non-clustered FedAvg baseline. |
| `gpmpc.py` | Sparse Gaussian-process model and GP-MPC baseline. |
| `metrics.py` | Evaluation metrics, aggregation utilities, and result-table generation. |
| `main.py` | Main experiment entry point. |
| `replot.py` | Replotting utilities for saved experiment outputs. |

## Requirements

The code is written in Python and mainly depends on:

- `numpy`
- `torch`
- `matplotlib`
- `pandas`

A typical setup is:

```bash
pip install numpy torch matplotlib pandas
```

## Usage

Run the full experiment:

```bash
python main.py
```

Regenerate plots from saved outputs:

```bash
python replot.py
```

By default, generated artifacts are written under:

```text
results/
  figures/
  tables/
  models/
```

## Methods Included

The implementation includes:

- Linear MPC
- Robust MPC
- FedAvg MPC
- CARRL-style risk-field MPC
- GP-MPC
- CFed-DSMPC (proposed)

## Notes

Only the Python source files are included in this repository. Generated figures, tables, model checkpoints, and local backup folders are intentionally excluded.
