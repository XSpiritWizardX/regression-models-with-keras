#!/usr/bin/env python3
"""
Quick visualizations for the concrete strength dataset.

Generates:
- histograms for all columns
- correlation heatmap
- scatter plots of each predictor vs target (Strength)

Images are saved to the specified output directory (default: figures/).
"""
from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")  # headless backend for saving to files
import matplotlib.pyplot as plt
import pandas as pd

DATA_URL = (
    "https://s3-api.us-geo.objectstorage.softlayer.net/"
    "cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv"
)
TARGET_COL = "Strength"


def load_data(data_url: str = DATA_URL) -> pd.DataFrame:
    return pd.read_csv(data_url)


def plot_histograms(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for ax, col in zip(axes.flatten(), df.columns):
        df[col].plot(kind="hist", bins=30, ax=ax, color="#4C72B0")
        ax.set_title(col)
        ax.set_ylabel("Count")
    fig.suptitle("Concrete dataset distributions", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(outdir / "histograms.png", dpi=150)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, outdir: Path) -> None:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(outdir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_scatter_targets(df: pd.DataFrame, outdir: Path) -> None:
    predictors = [c for c in df.columns if c != TARGET_COL]
    rows, cols = 4, 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 16))
    for ax, col in zip(axes.flatten(), predictors):
        ax.scatter(df[col], df[TARGET_COL], alpha=0.6, s=15, color="#1B9E77")
        ax.set_xlabel(col)
        ax.set_ylabel(TARGET_COL)
    fig.suptitle("Predictors vs Strength", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(outdir / "predictors_vs_strength.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate basic visualizations for the concrete dataset."
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory to save plots (default: figures)",
    )
    parser.add_argument(
        "--data-url",
        default=DATA_URL,
        help="Dataset URL or local CSV path (default: course dataset URL)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data_url)
    plot_histograms(df, outdir)
    plot_correlation(df, outdir)
    plot_scatter_targets(df, outdir)

    print(f"Saved visualizations to {outdir}/")


if __name__ == "__main__":
    main()
