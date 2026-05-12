"""Data loaders: CSV loading and train/test splitting."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logging import get_logger


_logger = get_logger(__name__)


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    df = pd.read_csv(path)
    _logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.15, random_state: int = 42):
    """Split DataFrame into features and target, then into train/test.

    Returns (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    _logger.info(f"Split: train={len(X_train)} rows, test={len(X_test)} rows")
    return X_train, X_test, y_train, y_test