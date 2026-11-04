"""
RailPulse AI - Phase 9 MLOps & Production Containerization Tests
Validates experiment tracking, DVC pipeline configuration, and Docker containerization assets.
"""
from pathlib import Path
import pytest
import yaml
from src.mlops.experiment_tracker import RailPulseExperimentTracker, compute_file_sha256


@pytest.fixture
def temp_tracker(tmp_path):
    log_dir = tmp_path / "experiments"
    return RailPulseExperimentTracker(experiment_dir=log_dir)


class TestPhase9MLOps:
    """Test suite for MLOps experiment tracking and artifact lineage."""

    def test_experiment_tracker_log_and_get(self, temp_tracker):
        run = temp_tracker.log_run(
            run_name="ST_GNN_v1.0",
            metrics={"mae": 1.42, "rmse": 2.15, "r2": 0.88, "wape": 0.12},
            params={"lr": 0.001, "hidden_dim": 64, "dropout": 0.2},
            tags={"phase": "phase9", "stage": "production"}
        )
        assert run["run_id"] is not None
        assert run["run_name"] == "ST_GNN_v1.0"
        assert run["metrics"]["mae"] == 1.42
        assert run["parameters"]["hidden_dim"] == 64

        retrieved = temp_tracker.get_run_by_id(run["run_id"])
        assert retrieved is not None
        assert retrieved["run_id"] == run["run_id"]

        latest = temp_tracker.get_latest_run()
        assert latest is not None
        assert latest["run_id"] == run["run_id"]

        all_runs = temp_tracker.list_runs()
        assert len(all_runs) == 1

    def test_file_hashing_and_tracking(self, temp_tracker, tmp_path):
        dummy_model = tmp_path / "model.pth"
        dummy_model.write_text("dummy model binary content", encoding="utf-8")

        h = compute_file_sha256(dummy_model)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 length

        run = temp_tracker.log_run(
            run_name="ST_GNN_Hashed",
            params={},
            metrics={"mae": 1.2},
            artifacts={"model_weights": dummy_model}
        )
        assert "model_weights" in run["artifacts"]
        assert run["artifacts"]["model_weights"]["sha256"] == h

    def test_dvc_pipeline_configuration(self):
        dvc_path = Path("dvc.yaml")
        assert dvc_path.exists(), "dvc.yaml pipeline configuration does not exist"
        
        with open(dvc_path, "r", encoding="utf-8") as f:
            dvc_content = yaml.safe_load(f)

        assert "stages" in dvc_content
        stages = dvc_content["stages"]
        assert "prepare_data" in stages
        assert "train_stgnn" in stages
        assert "cmd" in stages["prepare_data"]
        assert "outs" in stages["train_stgnn"]

        dvcignore_path = Path(".dvcignore")
        assert dvcignore_path.exists(), ".dvcignore file does not exist"

    def test_docker_configuration_files(self):
        backend_docker = Path("docker/Dockerfile.backend")
        frontend_docker = Path("docker/Dockerfile.frontend")
        nginx_conf = Path("docker/nginx.conf")
        docker_compose = Path("docker-compose.yml")

        assert backend_docker.exists(), "Dockerfile.backend is missing"
        assert frontend_docker.exists(), "Dockerfile.frontend is missing"
        assert nginx_conf.exists(), "nginx.conf is missing"
        assert docker_compose.exists(), "docker-compose.yml is missing"

        with open(docker_compose, "r", encoding="utf-8") as f:
            compose_data = yaml.safe_load(f)

        assert "services" in compose_data
        services = compose_data["services"]
        assert "backend" in services
        assert "frontend" in services
        assert "postgis" in services
        assert services["backend"]["ports"] == ["8001:8001"]
        assert services["frontend"]["ports"] == ["8000:8000"]
