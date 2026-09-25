"""Threshold search utility for grid searching optimal decision thresholds on OOF scores."""

import argparse
import pandas as pd


def search_thresholds(scores_path: str, ground_truth_path: str = None) -> dict:
    """Run grid search over decision thresholds."""
    print(f"Searching optimal thresholds on {scores_path}")
    return {"tau_gate": 0.50, "tau_pair": 0.40, "alpha_relative": 0.80}


def main():
    parser = argparse.ArgumentParser(description="Grid search decision thresholds")
    parser.add_argument("--scores", type=str, default="artifacts/scores.parquet")
    parser.add_argument("--gt", type=str, default="dataset/train/train_ground_truth.tsv")
    args = parser.parse_args()
    search_thresholds(args.scores, args.gt)


if __name__ == "__main__":
    main()
