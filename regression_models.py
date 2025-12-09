#!/usr/bin/env python3
"""
Keras regression models for the concrete strength dataset.

This script mirrors the walkthrough from the course material without relying on
notebooks. It loads the concrete data, normalizes the predictors, and trains
either a simple model with two hidden layers or a deeper variant with five
hidden layers. Use --help for CLI options.
"""
import argparse
import os
from typing import Iterable, Optional

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

DATA_URL = (
    "https://s3-api.us-geo.objectstorage.softlayer.net/"
    "cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv"
)


def load_dataset(data_url: str = DATA_URL) -> tuple[pd.DataFrame, pd.Series]:
    """Download the concrete dataset and return normalized predictors and target."""
    concrete_data = pd.read_csv(data_url)
    predictors = concrete_data.loc[:, concrete_data.columns != "Strength"]
    target = concrete_data["Strength"]
    predictors_norm = (predictors - predictors.mean()) / predictors.std()
    return predictors_norm, target


def build_model(n_features: int, hidden_units: Iterable[int]) -> tf.keras.Model:
    """
    Build a compiled Keras regression model.

    Args:
        n_features: Number of input features.
        hidden_units: Iterable with the number of units per hidden layer.
    """
    model = Sequential()
    model.add(Input(shape=(n_features,)))
    for units in hidden_units:
        model.add(Dense(units, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(
    model_type: str,
    epochs: int,
    val_split: float,
    data_url: str,
    batch_size: Optional[int],
) -> tf.keras.callbacks.History:
    """Train either the base (2-layer) or deep (5-layer) regression model."""
    predictors_norm, target = load_dataset(data_url)
    n_cols = predictors_norm.shape[1]
    hidden_map = {
        "base": [50, 50],
        "deep": [50, 50, 50, 50, 50],
    }
    model = build_model(n_cols, hidden_map[model_type])
    history = model.fit(
        predictors_norm,
        target,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Keras regression models on the concrete strength dataset."
    )
    parser.add_argument(
        "--model",
        choices=["base", "deep"],
        default="base",
        help="Which architecture to train: "
        "'base' has two hidden layers; 'deep' has five hidden layers.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.3,
        help="Fraction of data used for validation (default: 0.3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size; defaults to Keras' automatic choice.",
    )
    parser.add_argument(
        "--data-url",
        default=DATA_URL,
        help="Override the dataset URL if you have a local copy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = train_model(
        model_type=args.model,
        epochs=args.epochs,
        val_split=args.val_split,
        data_url=args.data_url,
        batch_size=args.batch_size,
    )
    losses = history.history.get("loss", [])
    val_losses = history.history.get("val_loss", [])

    if losses:
        print(f"Training complete. Final loss: {losses[-1]:.4f}")
    if val_losses:
        best_val_epoch = int(np.argmin(val_losses)) + 1
        best_val_loss = float(val_losses[best_val_epoch - 1])
        print(f"Final val_loss: {val_losses[-1]:.4f}")
        print(f"Best val_loss: {best_val_loss:.4f} at epoch {best_val_epoch}")


if __name__ == "__main__":
    main()
