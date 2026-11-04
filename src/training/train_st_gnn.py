"""Master training & benchmarking script for Phase 4 ST-GNN (RailPulse AI).

Loads PyG DataLoaders from the processed features NPZ, trains the
RailPulseSTGNN with mixed precision, evaluates on the test set, and
prints a side-by-side comparison against Phase 3 classical baselines.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR, TARGET_STATION_EVAS, DEVICE
from src.graph_construction.static_graph_builder import RailwayGraphBuilder
from src.graph_construction.temporal_graph_dataset import (
    RailwayGraphDataset,
    create_pyg_dataloaders,
)
from src.models.st_gnn import RailPulseSTGNN
from src.training.st_gnn_trainer import STGNNTrainer
from src.training.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

NPZ_PATH = PROCESSED_DIR / "frankfurt_corridor_features.npz"
ST_GNN_RESULTS_PATH = PROCESSED_DIR / "st_gnn_benchmark_results.json"
PHASE3_INITIAL_PATH = PROCESSED_DIR / "baseline_benchmark_results.json"
PHASE3_ENHANCED_PATH = PROCESSED_DIR / "improved_baseline_benchmark_results.json"
CHECKPOINT_PATH = Path("models/st_gnn_best.pth")

HORIZON_NAMES = ["15m", "30m", "60m"]
METRIC_NAMES = ["mae", "rmse", "r2", "wape"]

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2


def build_dataloaders() -> tuple:
    """Build PyG DataLoaders from the processed features NPZ."""
    builder = RailwayGraphBuilder(target_evas=list(TARGET_STATION_EVAS))
    builder.build()
    logger.info(
        f"Static graph: {builder.num_nodes} nodes, {builder.edge_index.shape[1]} edges"
    )

    dataset = RailwayGraphDataset(NPZ_PATH, builder)
    logger.info(f"Dataset: {len(dataset)} snapshots, {dataset.num_nodes} nodes")

    train_loader, val_loader, test_loader = create_pyg_dataloaders(
        dataset, batch_size=BATCH_SIZE, train_ratio=0.70, val_ratio=0.15
    )
    return train_loader, val_loader, test_loader, dataset.num_nodes, builder.edge_index.shape[1]


def load_phase3_results() -> Dict:
    """Load Phase 3 initial and enhanced results for comparison."""
    phase3 = {"initial": {}, "enhanced": {}}
    for key, path in (("initial", PHASE3_INITIAL_PATH), ("enhanced", PHASE3_ENHANCED_PATH)):
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            phase3[key] = data.get("models", {})
        else:
            logger.warning(f"Phase 3 results not found: {path}")
    return phase3


def print_side_by_side_table(st_gnn_metrics: Dict, phase3: Dict) -> None:
    """Print side-by-side comparison: Phase 3 baselines vs Phase 4 ST-GNN."""
    horizons = HORIZON_NAMES + ["overall"]
    enhanced = phase3.get("enhanced", {})
    initial = phase3.get("initial", {})

    print("\n" + "=" * 105)
    print("      PHASE 3 BASELINES (Persistence, XGBoost, LightGBM) vs PHASE 4 ST-GNN (Test Benchmark)")
    print("=" * 105)
    header = f"{'Horizon':<8} | {'Model':<25} | {'Paradigm':<20} | {'MAE (min)':>10} | {'RMSE (min)':>10} | {'R²':>8} | {'WAPE':>8}"
    print(header)
    print("-" * 105)

    for h in horizons:
        rows = []

        # 1. Naive Persistence
        p_m = enhanced.get("naive_persistence", {}).get(h) or initial.get("naive_persistence", {}).get(h, {})
        if p_m:
            rows.append(("Naive Persistence", "Phase 3 Baseline", p_m))

        # 2. XGBoost
        xgb_m = enhanced.get("xgboost", {}).get(h) or initial.get("xgboost", {}).get(h, {})
        if xgb_m:
            rows.append(("XGBoost", "Phase 3 Tabular", xgb_m))

        # 3. LightGBM
        lgb_m = enhanced.get("lightgbm", {}).get(h) or initial.get("lightgbm", {}).get(h, {})
        if lgb_m:
            rows.append(("LightGBM", "Phase 3 Tabular", lgb_m))

        # 4. Phase 4 ST-GNN
        st_m = st_gnn_metrics.get(h, {})
        if st_m:
            rows.append(("RailPulse ST-GNN", "Phase 4 Spatio-Temporal", st_m))

        for idx, (m_name, m_type, m_vals) in enumerate(rows):
            h_label = h if idx == 0 else ""
            mae = m_vals.get("mae", float("nan"))
            rmse = m_vals.get("rmse", float("nan"))
            r2 = m_vals.get("r2", float("nan"))
            wape = m_vals.get("wape", float("nan"))

            mae_str = f"{mae:10.4f}" if np.isfinite(mae) else "       n/a"
            rmse_str = f"{rmse:10.4f}" if np.isfinite(rmse) else "       n/a"
            r2_str = f"{r2:8.4f}" if np.isfinite(r2) else "     n/a"
            wape_str = f"{wape:8.4f}" if np.isfinite(wape) else "     n/a"

            print(f"{h_label:<8} | {m_name:<25} | {m_type:<20} | {mae_str} | {rmse_str} | {r2_str} | {wape_str}")

        print("-" * 105)
    print("=" * 105 + "\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger.info(f"Device: {DEVICE.type} ({DEVICE})")

    # 1. Build PyG DataLoaders
    train_loader, val_loader, test_loader, num_nodes, num_edges = build_dataloaders()

    # 2. Instantiate ST-GNN model
    model = RailPulseSTGNN(
        node_in_dim=11,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        out_horizons=3,
        dropout=DROPOUT,
    )
    logger.info(
        f"Model: {num_nodes} nodes, hidden_dim={HIDDEN_DIM}, layers={NUM_LAYERS}, "
        f"dropout={DROPOUT}"
    )

    # 3. Train with mixed precision
    trainer = STGNNTrainer(
        model=model,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device=str(DEVICE),
    )
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        save_path=CHECKPOINT_PATH,
    )

    # 4. Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader)
    logger.info(
        f"Test loss={test_loss:.4f} | test_mae={test_metrics['overall']['mae']:.4f} | "
        f"test_r2={test_metrics['overall']['r2']:.4f}"
    )

    # 5. Save results
    results: Dict = {
        "data": {
            "n_samples": len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
            "n_nodes": num_nodes,
            "n_edges": num_edges,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
        },
        "model": {
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        },
        "training": {
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "test_loss": test_loss,
        },
        "models": {"st_gnn": test_metrics},
    }

    ST_GNN_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ST_GNN_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"ST-GNN results saved to {ST_GNN_RESULTS_PATH}")

    # 6. Side-by-side comparison with Phase 3
    phase3 = load_phase3_results()
    print_side_by_side_table(test_metrics, phase3)


if __name__ == "__main__":
    main()
