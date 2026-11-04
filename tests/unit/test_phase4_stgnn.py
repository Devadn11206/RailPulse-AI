"""Phase 4 ST-GNN diagnostic tests."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.st_gnn import RailPulseSTGNN
from src.training.st_gnn_trainer import STGNNTrainer


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(num_nodes: int = 10, num_edges: int = 20) -> Data:
    """Create a synthetic PyG Data object."""
    x = torch.randn(num_nodes, 88, dtype=torch.float32)  # 4 static + 84 temporal
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    edge_attr = torch.rand(num_edges, 1, dtype=torch.float32) * 60.0
    y = torch.randn(num_nodes, 3, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _make_synthetic_loader(num_samples: int = 4, num_nodes: int = 10,
                           num_edges: int = 20, batch_size: int = 2) -> DataLoader:
    """Create a synthetic PyG DataLoader."""
    data_list = [_make_synthetic_data(num_nodes, num_edges) for _ in range(num_samples)]
    return DataLoader(data_list, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# RailPulseSTGNN tests
# ---------------------------------------------------------------------------

def test_stgnn_forward_output_shape_and_dtype():
    """Forward pass on synthetic PyG Data yields (Num_Nodes, 3) float32."""
    model = RailPulseSTGNN(node_in_dim=11, hidden_dim=32, num_layers=2, out_horizons=3, dropout=0.2)
    model.eval()
    data = _make_synthetic_data(num_nodes=10, num_edges=20)
    with torch.no_grad():
        out = model(data)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (10, 3), f"Expected (10, 3), got {out.shape}"
    assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"


def test_stgnn_forward_with_batch():
    """Forward pass on batched PyG Data yields correct output shape."""
    model = RailPulseSTGNN(node_in_dim=11, hidden_dim=32, num_layers=2, out_horizons=3, dropout=0.2)
    model.eval()
    data_list = [_make_synthetic_data(num_nodes=10, num_edges=20) for _ in range(3)]
    batch = Batch.from_data_list(data_list)
    with torch.no_grad():
        out = model(batch)
    total_nodes = 10 * 3
    assert out.shape == (total_nodes, 3), f"Expected ({total_nodes}, 3), got {out.shape}"
    assert out.dtype == torch.float32


def test_stgnn_different_hidden_dims():
    """Model works with different hidden dimensions (<=64 per safety constraint)."""
    for hidden_dim in (16, 32, 64):
        model = RailPulseSTGNN(node_in_dim=11, hidden_dim=hidden_dim, num_layers=2, out_horizons=3)
        model.eval()
        data = _make_synthetic_data(num_nodes=10, num_edges=20)
        with torch.no_grad():
            out = model(data)
        assert out.shape == (10, 3)
        assert out.dtype == torch.float32


def test_stgnn_train_mode_gradients_flow():
    """Model produces gradients in train mode."""
    model = RailPulseSTGNN(node_in_dim=11, hidden_dim=32, num_layers=2, out_horizons=3)
    model.train()
    data = _make_synthetic_data(num_nodes=10, num_edges=20)
    out = model(data)
    loss = out.sum()
    loss.backward()
    # Check that at least some parameters have gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients flowed through the model"


# ---------------------------------------------------------------------------
# STGNNTrainer tests
# ---------------------------------------------------------------------------

def test_trainer_train_eval_finite_loss():
    """One train step and one eval step produce finite loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RailPulseSTGNN(node_in_dim=11, hidden_dim=32, num_layers=2, out_horizons=3)
    trainer = STGNNTrainer(model=model, lr=1e-3, weight_decay=1e-4, device=device)

    train_loader = _make_synthetic_loader(num_samples=4, batch_size=2)
    val_loader = _make_synthetic_loader(num_samples=2, batch_size=2)

    train_loss = trainer.train_epoch(train_loader)
    assert np.isfinite(train_loss), f"Train loss not finite: {train_loss}"

    val_loss, val_metrics = trainer.evaluate(val_loader)
    assert np.isfinite(val_loss), f"Val loss not finite: {val_loss}"
    assert "overall" in val_metrics
    assert "mae" in val_metrics["overall"]


def test_trainer_parameter_gradients_update():
    """Model parameters change after a training step (gradients update)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RailPulseSTGNN(node_in_dim=11, hidden_dim=32, num_layers=2, out_horizons=3)
    trainer = STGNNTrainer(model=model, lr=1e-3, weight_decay=1e-4, device=device)

    train_loader = _make_synthetic_loader(num_samples=4, batch_size=2)

    # Capture parameters before training
    params_before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    trainer.train_epoch(train_loader)

    # Check that at least some parameters changed
    changed = False
    for k in params_before:
        if not torch.equal(params_before[k], model.state_dict()[k]):
            changed = True
            break
    assert changed, "Model parameters did not update after train_epoch"


def test_trainer_early_stopping_state():
    """Trainer initializes early stopping state correctly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RailPulseSTGNN(node_in_dim=11, hidden_dim=32, num_layers=2, out_horizons=3)
    trainer = STGNNTrainer(model=model, lr=1e-3, device=device)
    assert trainer.best_val_loss == float("inf")
    assert trainer.patience == 10
    assert trainer.patience_counter == 0
    assert trainer.best_state_dict is None
