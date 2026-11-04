"""Enhanced classical baseline benchmark trainer for RailPulse AI Phase 3.

Uses engineered tabular features (spatial lag differentials, network rolling
momentum, cyclical interactions, station identity) and per-horizon single-output
tree models with tuned hyperparameters. Saves results to
data/processed/improved_baseline_benchmark_results.json and prints a side-by-side
comparison against the initial Phase 3 results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.config import PROCESSED_DIR
from src.feature_engineering.enhanced_tabular_features import (
    EnhancedTabularFeatureBuilder,
    build_enhanced_features,
)
from src.models.baselines import NaivePersistenceBaseline, TreeBaselineSuite
from src.training.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

NPZ_PATH = PROCESSED_DIR / "frankfurt_corridor_features.npz"
RESULTS_PATH = PROCESSED_DIR / "improved_baseline_benchmark_results.json"
INITIAL_RESULTS_PATH = PROCESSED_DIR / "baseline_benchmark_results.json"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RANDOM_STATE = 42
HORIZON_NAMES = ["15m", "30m", "60m"]


def load_and_flatten(npz_path: Path = NPZ_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Load NPZ and flatten X from [N, 12, 7] to [N, 84]."""
    data = np.load(npz_path, allow_pickle=False)
    X = data["X"].astype(np.float32)
    Y = data["Y"].astype(np.float32)
    N = X.shape[0]
    X_flat = X.reshape(N, -1).astype(np.float32)  # [N, 84]
    return X_flat, Y


def chronological_split(X: np.ndarray, Y: np.ndarray,
                        train_ratio: float = TRAIN_RATIO,
                        val_ratio: float = VAL_RATIO) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    splits = {
        "train": (X[:train_end], Y[:train_end]),
        "val": (X[train_end:val_end], Y[train_end:val_end]),
        "test": (X[val_end:], Y[val_end:]),
    }
    return splits


def train_and_evaluate() -> Dict:
    evaluator = ModelEvaluator()
    X_flat, Y = load_and_flatten()
    splits = chronological_split(X_flat, Y)

    X_train_flat, Y_train = splits["train"]
    X_val_flat, Y_val = splits["val"]
    X_test_flat, Y_test = splits["test"]

    logger.info(f"Train: {X_train_flat.shape}, Val: {X_val_flat.shape}, Test: {X_test_flat.shape}")

    # Build enhanced features for tree models
    builder = EnhancedTabularFeatureBuilder(num_stations=10)
    X_train_enh = builder.build(X_train_flat)
    X_val_enh = builder.build(X_val_flat)
    X_test_enh = builder.build(X_test_flat)

    logger.info(f"Enhanced feature dim: {X_train_enh.shape[1]}")

    results: Dict = {
        "data": {
            "n_samples": int(len(X_flat)),
            "n_features_flat": int(X_flat.shape[1]),
            "n_features_enhanced": int(X_train_enh.shape[1]),
            "n_targets": int(Y.shape[1]),
            "train_samples": int(len(X_train_flat)),
            "val_samples": int(len(X_val_flat)),
            "test_samples": int(len(X_test_flat)),
        },
        "models": {},
    }

    # Naive Persistence (evaluated on enhanced features, reads col 77)
    logger.info("Training NaivePersistenceBaseline...")
    naive = NaivePersistenceBaseline()
    naive.fit(X_train_enh, Y_train)
    Y_pred_naive = naive.predict(X_test_enh)
    results["models"]["naive_persistence"] = evaluator.evaluate_multi_horizon(Y_test, Y_pred_naive)

    # Enhanced tree baselines (per-horizon, on enhanced features)
    for name, model in TreeBaselineSuite.all_models(random_state=RANDOM_STATE).items():
        logger.info(f"Training enhanced {name}...")
        model.fit(X_train_enh, Y_train)
        Y_pred = model.predict(X_test_enh)
        results["models"][name] = evaluator.evaluate_multi_horizon(Y_test, Y_pred)

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Enhanced results saved to {RESULTS_PATH}")

    # Print side-by-side comparison with initial results
    print_enhanced_comparison_table(results, INITIAL_RESULTS_PATH)

    return results


def print_enhanced_comparison_table(
    enhanced_results: Dict, initial_results_path: Path
) -> None:
    """Print a side-by-side metric comparison: Initial vs Enhanced Phase 3."""
    horizons = ["15m", "30m", "60m", "overall"]
    metrics = ["mae", "rmse", "r2", "wape"]

    # Load initial results if available
    initial_models: Dict = {}
    if initial_results_path.exists():
        with open(initial_results_path) as f:
            initial_results = json.load(f)
        initial_models = initial_results.get("models", {})

    print("\n" + "=" * 110)
    print("PHASE 3 BASELINE COMPARISON: INITIAL vs ENHANCED (Test Set)")
    print("=" * 110)

    # Header
    col_groups = []
    for version in ("Initial", "Enhanced"):
        for h in horizons:
            for m in metrics:
                col_groups.append(f"{version} {h} {m.upper()}")
    header = "| Model | " + " | ".join(col_groups) + " |"
    sep = "|---|" + "|".join("----:" for _ in col_groups) + "|"
    print(header)
    print(sep)

    all_model_names = list(enhanced_results["models"].keys())

    for model_name in all_model_names:
        row = f"| {model_name} |"
        enhanced_m = enhanced_results["models"][model_name]
        initial_m = initial_models.get(model_name, {})

        for h in horizons:
            for metric in metrics:
                # Enhanced value
                e_val = enhanced_m.get(h, {}).get(metric, 0.0)
                # Initial value
                i_val = initial_m.get(h, {}).get(metric, float("nan"))
                if np.isnan(i_val):
                    row += f" {e_val:.4f} | n/a |"
                else:
                    row += f" {i_val:.4f} | {e_val:.4f} |"
        print(row)

    print()
    print("Legend: Initial | Enhanced  (per model, per horizon, per metric)")
    print("=" * 110 + "\n")

    # Also print enhanced-only detailed table for readability
    print_enhanced_only_table(enhanced_results)


def print_enhanced_only_table(results: Dict) -> None:
    """Print enhanced-only detailed table for readability."""
    horizons = ["15m", "30m", "60m", "overall"]
    metrics = ["mae", "rmse", "r2", "wape"]

    print("\n" + "=" * 70)
    print("ENHANCED PHASE 3 BASELINE RESULTS (Test Set)")
    print("=" * 70)

    header = "| Model | " + " | ".join(f"{h} {m.upper()}" for h in horizons for m in metrics) + " |"
    sep = "|---|" + "|".join("----:" for _ in horizons * len(metrics)) + "|"
    print(header)
    print(sep)

    for model_name, model_results in results["models"].items():
        row = f"| {model_name} |"
        for h in horizons:
            m = model_results.get(h, {})
            for metric in metrics:
                val = m.get(metric, 0.0)
                row += f" {val:.4f} |"
        print(row)
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_and_evaluate()
