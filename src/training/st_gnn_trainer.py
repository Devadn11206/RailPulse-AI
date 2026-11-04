"""PyTorch mixed-precision trainer for RailPulse ST-GNN (Phases 4 & 10).

Implements the STGNNTrainer with:
  - AdamW optimizer + HuberLoss (Smooth L1) for outlier-robust training
  - torch.amp.GradScaler for 16-bit/32-bit mixed precision (bounded VRAM < 1.5 GB)
  - Memory-bounded dynamic subgraph loader (NeighborLoader with RandomNodeLoader fallback)
  - Early stopping with patience=10 based on validation loss
  - Per-horizon metrics (MAE, RMSE, R2, WAPE) via ModelEvaluator
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader, RandomNodeLoader

from src.training.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class STGNNTrainer:
    """Mixed-precision trainer for the RailPulseSTGNN model."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.HuberLoss(delta=1.0)
        self.scaler = GradScaler("cuda", enabled=(self.device.type == "cuda"))
        self.evaluator = ModelEvaluator()

        # Early stopping state
        self.best_val_loss = float("inf")
        self.patience = 10
        self.patience_counter = 0
        self.best_state_dict: Optional[dict] = None

    @staticmethod
    def create_subgraph_loader(
        data: Data,
        batch_size: int = 16,
        num_neighbors: Optional[List[int]] = None,
        shuffle: bool = True,
    ) -> Union[NeighborLoader, RandomNodeLoader]:
        """Create a memory-bounded PyG subgraph loader.
        Uses NeighborLoader if C++ kernels (pyg-lib/torch-sparse) are installed;
        otherwise falls back seamlessly to RandomNodeLoader for cross-platform zero-dependency bounding.
        """
        import torch_geometric.typing as tgt

        if tgt.WITH_PYG_LIB or tgt.WITH_TORCH_SPARSE:
            if num_neighbors is None:
                num_neighbors = [15, 10]
            return NeighborLoader(
                data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                shuffle=shuffle,
            )
        else:
            num_nodes = data.num_nodes or data.x.shape[0]
            num_parts = max(1, int(np.ceil(num_nodes / max(1, batch_size))))
            return RandomNodeLoader(
                data,
                num_parts=num_parts,
                shuffle=shuffle,
            )

    @classmethod
    def create_neighbor_loader(
        cls,
        data: Data,
        num_neighbors: Optional[List[int]] = None,
        batch_size: int = 16,
        shuffle: bool = True,
    ) -> Union[NeighborLoader, RandomNodeLoader]:
        """Alias for backward-compatible Phase 10 API."""
        return cls.create_subgraph_loader(
            data=data,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
            shuffle=shuffle,
        )

    def train_epoch(self, dataloader: Union[DataLoader, NeighborLoader, RandomNodeLoader, Iterable]) -> float:
        """Execute one training epoch. Returns average train loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                output = self.model(batch)  # [N_batch, 3]
                target = batch.y
                # If NeighborLoader sampled batch_size seeds, slice seed targets if available
                if hasattr(batch, "batch_size") and batch.batch_size is not None and batch.batch_size < output.shape[0]:
                    loss = self.criterion(output[:batch.batch_size], target[:batch.batch_size])
                else:
                    loss = self.criterion(output, target)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.detach().cpu())
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, dataloader: Union[DataLoader, NeighborLoader, RandomNodeLoader, Iterable]) -> Tuple[float, Dict[str, Dict[str, float]]]:
        """Evaluate on a PyG DataLoader or subgraph loader. Returns (avg_loss, metrics_dict)."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_targets = []
        all_predictions = []

        for batch in dataloader:
            batch = batch.to(self.device)

            with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                output = self.model(batch)
                target = batch.y

                if hasattr(batch, "batch_size") and batch.batch_size is not None and batch.batch_size < output.shape[0]:
                    out_eval = output[:batch.batch_size]
                    tgt_eval = target[:batch.batch_size]
                else:
                    out_eval = output
                    tgt_eval = target

                loss = self.criterion(out_eval, tgt_eval)

            total_loss += float(loss.detach().cpu())
            num_batches += 1

            all_targets.append(tgt_eval.detach().cpu().float().numpy())
            all_predictions.append(out_eval.detach().cpu().float().numpy())

        avg_loss = total_loss / max(num_batches, 1)

        y_true = np.vstack(all_targets) if all_targets else np.zeros((0, 3))
        y_pred = np.vstack(all_predictions) if all_predictions else np.zeros((0, 3))

        metrics = self.evaluator.evaluate_multi_horizon(y_true, y_pred)
        return avg_loss, metrics

    def fit(
        self,
        train_loader: Union[DataLoader, NeighborLoader, RandomNodeLoader],
        val_loader: Union[DataLoader, NeighborLoader, RandomNodeLoader],
        epochs: int = 50,
        save_path: Path = Path("models/st_gnn_best.pth"),
    ) -> Dict:
        """Complete training loop with validation, early stopping, checkpointing."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        history = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": 0,
            "best_val_loss": float("inf"),
        }

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            logger.info(
                f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_mae={val_metrics['overall']['mae']:.4f} | "
                f"val_r2={val_metrics['overall']['r2']:.4f}"
            )

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                history["best_epoch"] = epoch
                history["best_val_loss"] = val_loss

                # Save best checkpoint
                torch.save(self.best_state_dict, save_path)
                logger.info(f"  -> New best model saved to {save_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(patience={self.patience}, best_val_loss={self.best_val_loss:.4f})"
                    )
                    break

        # Restore best weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        history["final_val_metrics"] = val_metrics
        return history
