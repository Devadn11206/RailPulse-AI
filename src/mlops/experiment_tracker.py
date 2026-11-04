"""MLOps Experiment Tracking & Artifact Lineage for RailPulse AI.

Implements lightweight, zero-dependency experiment logging compatible with
standard MLflow/DVC workflows. Records model hyperparameters, dataset hashes,
multi-horizon evaluation metrics, and system environment details.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of an artifact file."""
    if not file_path.exists():
        return "file_not_found"
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class RailPulseExperimentTracker:
    """Logs and queries ML training and evaluation experiment runs."""

    def __init__(self, experiment_dir: Optional[Path] = None):
        if experiment_dir is None:
            # Default to <BASE_DIR>/experiments
            base_dir = Path(__file__).resolve().parents[2]
            self.exp_dir = base_dir / "experiments"
        else:
            self.exp_dir = Path(experiment_dir)

        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.runs_file = self.exp_dir / "runs.json"
        self._runs: List[Dict[str, Any]] = self._load_runs()

    def _load_runs(self) -> List[Dict[str, Any]]:
        if self.runs_file.exists():
            try:
                with open(self.runs_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not read runs.json: %s", e)
                return []
        return []

    def log_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        artifacts: Optional[Dict[str, Path]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Record an experiment execution run.

        Parameters
        ----------
        run_name:
            Descriptive run name (e.g. 'st_gnn_rtx3050_production').
        params:
            Model hyperparameters (lr, hidden_dim, dropout, etc.).
        metrics:
            Multi-horizon metrics (MAE, RMSE, R2, WAPE).
        artifacts:
            Dict of name -> file path to track with SHA-256 hashes.
        tags:
            Key-value annotations.

        Returns
        -------
        Complete logged run record dictionary.
        """
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"

        artifact_records: Dict[str, Dict[str, Any]] = {}
        if artifacts:
            for art_name, path in artifacts.items():
                p = Path(path)
                artifact_records[art_name] = {
                    "path": str(p),
                    "exists": p.exists(),
                    "size_bytes": p.stat().st_size if p.exists() else 0,
                    "sha256": compute_file_sha256(p) if p.exists() else None,
                }

        run_record: Dict[str, Any] = {
            "run_id": run_id,
            "run_name": run_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "python_version": platform.python_version(),
                "os": platform.system(),
                "machine": platform.machine(),
            },
            "parameters": params,
            "metrics": metrics,
            "artifacts": artifact_records,
            "tags": tags or {},
        }

        self._runs.append(run_record)
        with open(self.runs_file, "w", encoding="utf-8") as f:
            json.dump(self._runs, f, indent=2)

        logger.info("Logged experiment run %s (%s)", run_id, run_name)
        return run_record

    def get_latest_run(self) -> Optional[Dict[str, Any]]:
        """Return the most recently logged run."""
        return self._runs[-1] if self._runs else None

    def get_run_by_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Lookup run by ID."""
        for run in self._runs:
            if run.get("run_id") == run_id:
                return run
        return None

    def list_runs(self) -> List[Dict[str, Any]]:
        """Return all logged experiment records."""
        return list(self._runs)

