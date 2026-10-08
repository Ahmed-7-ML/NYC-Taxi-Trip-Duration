"""Model loading helpers for the prediction API."""

import os
import sys
from pathlib import Path
import mlflow
from mlflow import MlflowClient
import sqlite3
import pickle
import tempfile

# `model_loader.py` is at pipeline/src/api/model_loader.py, so the pipeline
# root (which contains `config/` and `src/`) is three levels up. 
# Putting it on sys.path lets us reuse the project-wide `from config.config import ...` style 
# from any working directory — this module is imported as a top-level module,
# so a relative import here can never work.
_PIPELINE_DIR = Path(__file__).resolve().parents[2]
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

# `config` is a top-level package living inside `pipeline/`, so this import must run *after* the sys.path bootstrap above.
from config.config import MlflowConfig

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{_PIPELINE_DIR}/mlflow_trip_duration.db",
)

# print(_PIPELINE_DIR)
# print(MLFLOW_TRACKING_URI)

MODEL_NAME = os.getenv("MODEL_NAME", "trip_duration_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
_ARTIFACTS_ROOT = os.getenv("MLFLOW_ARTIFACTS_ROOT")

class _State:
    model = None
    preprocessor = None
    version: str = 'Unknown'
    alias: str = MODEL_ALIAS

_state = _State()

def _remap(path: str) -> str:
    if not _ARTIFACTS_ROOT or not path:
        return path
    idx = path.find("/mlruns/")
    return _ARTIFACTS_ROOT + path[idx:] if idx >= 0 else path

def _get_storage_location(version:str) -> str:
    """
    """
    db_path = MLFLOW_TRACKING_URI.replace('sqlite:///', '')
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    row = cursor.execute(
        "SELECT storage_location FROM model_versions WHERE name=? AND version=?",
        (MODEL_NAME, version)
    ).fetchone()
    connection.close()
    return row[0] if row else row

def _get_artifact_uri(run_id:str) -> str:
    """
    """
    db_path = MLFLOW_TRACKING_URI.replace('sqlite:///', '')
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    row = cursor.execute(
        "SELECT artifact_uri FROM runs WHERE run_uuid=?", (run_id)
    ).fetchone()
    connection.close()
    return row[0] if row else None

def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    model = client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_ALIAS)
    _state.version = f"v{model.version}"
    _state.alias = MODEL_ALIAS
    
    if _ARTIFACTS_ROOT:
        storage_location = _get_storage_location(model.version)
        _state.model = mlflow.sklearn.load_model(storage_location)
    else:
        _state.model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    
    try:
        if _ARTIFACTS_ROOT:
            artifact_uri = _get_artifact_uri(model.run_id)
            preprocessor_path = Path(_remap(artifact_uri)) / "preprocessor" / "preprocessor.pkl"
            with open(preprocessor_path, "rb") as f:
                _state.preprocessor = pickle.load(f)
        else:
            dst = tempfile.mkdtemp()
            art_path = mlflow.artifacts.download_artifacts(
                run_id=model.run_id,
                artifact_path="preprocessor/preprocessor.pkl",
                dst_path=dst
            )
            with open(art_path, "rb") as f:
                _state.preprocessor = pickle.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Preprocessor not found in MLflow run {model.run_id[:8]}. "
            "Re-run the pipeline with --promote to register a model that includes it."
        ) from e

def get_State() -> _State:
    return _state
