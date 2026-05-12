"""Model registry: save / load pipelines via joblib."""
import os
from pathlib import Path
import joblib

from src.utils.logging import get_logger


_logger = get_logger(__name__)


def get_model_path(name: str = "churn_model", version: int = 1, models_dir: str = None) -> str:
    """Return a sanitized path under the models directory."""
    if models_dir is None:
        models_dir = Path(__file__).parent.parent.parent / "models"
    else:
        models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{name}_v{version}.joblib"
    return str(models_dir / filename)


def save_model(pipeline, path: str = None, name: str = "churn_model", version: int = 1) -> str:
    """Save a fitted pipeline to a .joblib file.

    Returns the path the model was saved to.
    """
    if path is None:
        path = get_model_path(name, version)
    joblib.dump(pipeline, path)
    _logger.info(f"Model saved to: {path}")
    return path


def load_model(path: str):
    """Load a saved .joblib pipeline."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    pipeline = joblib.load(path)
    _logger.info(f"Model loaded from: {path}")
    return pipeline