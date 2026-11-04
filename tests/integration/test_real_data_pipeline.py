"""Integration test for real corridor data pipeline."""

import numpy as np
import pytest

from pathlib import Path

NPZ_PATH = Path("data/processed/frankfurt_corridor_features.npz")


@pytest.mark.skipif(not NPZ_PATH.exists(), reason="Processed features not generated yet")
def test_real_corridor_npz():
    with np.load(NPZ_PATH) as data:
        assert "X" in data.files
        assert "Y" in data.files
        X = data["X"]
        Y = data["Y"]
    assert X.ndim == 3
    assert X.shape[1:] == (12, 7)
    assert Y.shape[1] == 3
    assert X.shape[0] == Y.shape[0]
    assert np.isfinite(X).all()
    assert np.isfinite(Y).all()
    assert not np.isnan(X).any()
    assert not np.isnan(Y).any()
