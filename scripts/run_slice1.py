"""CLI entrypoint for Slice 1: build 8-week temporal split and evaluate Popularity Baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.baselines.popularity import evaluate_popularity_baseline
from src.data.temporal_split import SplitConfig, build_temporal_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Slice 1 data split and popularity baseline.")
    parser.add_argument("--transactions", type=Path, required=True, help="Path to transactions_train.csv")
    parser.add_argument("--articles", type=Path, required=True, help="Path to articles.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Output directory")
    args = parser.parse_args()

    counts = build_temporal_split(args.transactions, args.articles, args.out_dir, SplitConfig())
    print("Split counts:", counts)

    val_metrics = evaluate_popularity_baseline(args.out_dir / "train.parquet", args.out_dir / "val.parquet")
    test_metrics = evaluate_popularity_baseline(args.out_dir / "train.parquet", args.out_dir / "test.parquet")
    print("Popularity Baseline (Val):", val_metrics)
    print("Popularity Baseline (Test):", test_metrics)


if __name__ == "__main__":
    main()
