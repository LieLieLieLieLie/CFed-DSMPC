"""
Rebuild figures from saved experiment outputs without retraining.

Examples:
    python replot.py --tag seed2
    python replot.py --tag seed2 --kind traj
    python replot.py --tag seed2 --kind summary
    python replot.py --tag seed2 --kind ood
    python replot.py --tag seed2 --kind ablation
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

from config import Config
from metrics import (
    plot_summary_bars,
    plot_per_cluster,
    plot_ood_heatmap,
    plot_federated_analysis,
    plot_ablation,
)
from utils import CrossingEnv, plot_experiment_1
from main import METHOD_NAMES, load_models


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _agent_keys_to_int(results):
    converted = {}
    for method, agents in results.items():
        converted[method] = {int(agent_id): record for agent_id, record in agents.items()}
    return converted


def _cluster_keys_to_int(collected_states):
    return {int(cid): states for cid, states in collected_states.items()}


def _load_metrics(tag):
    path = os.path.join(Config.TABLES_DIR, f"{tag}_metrics.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return _load_json(path)


def _load_federated_table(tag, prefix, value_field):
    pattern = os.path.join(Config.TABLES_DIR, f"{tag}_{prefix}*_summary.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing federated summary file matching: {pattern}")

    results = {}
    with open(matches[-1], "r", newline="") as f:
        for row in csv.DictReader(f):
            raw_value = row[value_field]
            value = int(raw_value) if value_field == "rounds" else float(raw_value)
            method = row["method"]
            results.setdefault(value, {})[method] = {
                "success_rate": float(row["success_rate"]),
                "safety_comply_rate": float(row["safety_comply_rate"]),
                "overall_min_dist_mean": float(row["overall_min_dist_mean"]),
                "corridor_correct_rate": float(row["corridor_correct_rate"]),
            }
    return results


def _load_trajectories(tag):
    path = os.path.join(Config.TABLES_DIR, f"{tag}_trajectories.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing trajectory file: {path}\n"
            "Run main.py once after the trajectory-saving changes, then replot."
        )
    payload = _load_json(path)
    return {
        "starts": [np.array(v, dtype=np.float32) for v in payload["starts"]],
        "targets": [np.array(v, dtype=np.float32) for v in payload["targets"]],
        "collected_states": _cluster_keys_to_int(payload["collected_states"]),
        "results": _agent_keys_to_int(payload["results"]),
    }


def replot_summary(tag):
    metrics = _load_metrics(tag)
    plot_summary_bars(metrics, tag)
    plot_per_cluster(metrics, tag)


def replot_traj(tag):
    data = _load_trajectories(tag)
    env = CrossingEnv(seed=Config.ENV_SEED)
    plot_experiment_1(
        env,
        data["results"],
        data["starts"],
        data["targets"],
        data["collected_states"],
        METHOD_NAMES,
        tag=tag,
    )


def replot_ood(tag):
    env = CrossingEnv(seed=Config.ENV_SEED)
    server, _, _, _ = load_models()
    plot_ood_heatmap(env, server, tag)


def replot_federated(tag):
    rounds = _load_federated_table(tag, "federated_rounds", "rounds")
    mu = _load_federated_table(tag, "federated_mu", "mu")
    plot_federated_analysis(rounds, mu, tag)


def replot_ablation(tag):
    path = os.path.join(Config.TABLES_DIR, f"{tag}_ablation.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing ablation file: {path}")
    plot_ablation(_load_json(path), f"{tag}_ablation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="seed2", help="Result tag, e.g. seed2")
    ap.add_argument(
        "--kind",
        default="all",
        choices=["all", "traj", "summary", "ood", "federated", "ablation"],
        help="Which figure set to rebuild.",
    )
    args = ap.parse_args()

    if args.kind in ("all", "summary"):
        replot_summary(args.tag)
    if args.kind in ("all", "traj"):
        replot_traj(args.tag)
    if args.kind in ("all", "ood"):
        replot_ood(args.tag)
    if args.kind in ("all", "federated"):
        replot_federated(args.tag)
    if args.kind in ("all", "ablation"):
        replot_ablation(args.tag)

    print(f"Replot done. Figures are in: {Config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
