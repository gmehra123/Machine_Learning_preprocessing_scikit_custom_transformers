"""Drift detection: PSI for numeric features, chi-squared for categorical."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.utils.logging import get_logger


_logger = get_logger(__name__)


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index (PSI) using quantile-based bucketing."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[-1] = breakpoints[-1] + 1e-6

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = expected_counts / (expected_counts.sum() + 1e-6)
    actual_pct = actual_counts / (actual_counts.sum() + 1e-6)

    psi_vals = (expected_pct - actual_pct) * np.log((expected_pct + 1e-6) / (actual_pct + 1e-6))
    return float(np.sum(psi_vals))


def detect_numeric_drift(train_series: pd.Series, new_series: pd.Series, threshold: float = 0.2) -> dict:
    """Detect drift in a numeric feature using PSI."""
    psi = compute_psi(train_series.dropna().values, new_series.dropna().values)
    return {"psi": psi, "drifted": psi > threshold}


def detect_categorical_drift(train_series: pd.Series, new_series: pd.Series, threshold: float = 0.05) -> dict:
    """Detect drift in a categorical feature using chi-squared test of proportions."""
    train_counts = train_series.value_counts().sort_index()
    new_counts = new_series.value_counts().reindex(train_counts.index, fill_value=0).sort_index()

    contingency = np.array([train_counts.values, new_counts.values])
    if contingency.shape[1] < 2:
        return {"chi2": 0.0, "p_value": 1.0, "drifted": False}

    chi2, p_value, _, _ = chi2_contingency(contingency)
    return {"chi2": float(chi2), "p_value": float(p_value), "drifted": p_value < threshold}


def run_drift_check(X_train: pd.DataFrame, X_new: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> dict:
    """Run drift detection on all specified features."""
    results = {}
    for col in numeric_cols:
        if col in X_train.columns and col in X_new.columns:
            results[col] = detect_numeric_drift(X_train[col], X_new[col])
    for col in categorical_cols:
        if col in X_train.columns and col in X_new.columns:
            results[col] = detect_categorical_drift(X_train[col], X_new[col])
    return results


def summarize_drift(drift_results: dict, psi_warning: float, psi_critical: float, chi2_threshold: float, logger=None) -> bool:
    """Summarize and log drift results. Returns True if any drift detected."""
    has_drift = False
    logger = logger or _logger
    for feature, result in drift_results.items():
        if "psi" in result:
            if result["psi"] > psi_critical:
                logger.warning(f"CRITICAL DRIFT | feature={feature} | psi={result['psi']:.4f} | threshold={psi_critical:.4f}")
                has_drift = True
            elif result["psi"] > psi_warning:
                logger.warning(f"Drift warning | feature={feature} | psi={result['psi']:.4f} | threshold={psi_warning:.4f}")
                has_drift = True
        if "p_value" in result:
            if result["drifted"]:
                logger.warning(f"CATEGORICAL DRIFT | feature={feature} | p_value={result['p_value']:.4f} | threshold={chi2_threshold:.4f}")
                has_drift = True
    if not has_drift:
        logger.info("No drift detected across all features.")
    return has_drift
