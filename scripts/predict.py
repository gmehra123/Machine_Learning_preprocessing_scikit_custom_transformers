"""Prediction entry point: load model, run drift check, output predictions."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.models.registry import load_model
from src.data.loaders import load_csv
from src.data.drift import run_drift_check, summarize_drift
from src.utils.config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    DRIFT_PSI_WARNING, DRIFT_PSI_CRITICAL,
)
from src.utils.logging import get_logger, log_metrics, log_pipeline_stage, log_drift_warning


_logger = get_logger(__name__)


def predict(model_path: str, data_path: str, output_path: str = None) -> pd.DataFrame:
    """Run predictions on incoming data with drift monitoring.

    Parameters
    ----------
    model_path : str
        Path to saved pipeline (.joblib).
    data_path : str
        Path to input CSV.
    output_path : str, optional
        If provided, write results to this path.

    Returns
    -------
    predictions DataFrame
    """
    log_pipeline_stage(_logger, "LOAD MODEL")
    pipeline = load_model(model_path)

    log_pipeline_stage(_logger, "LOAD DATA")
    df = load_csv(data_path)
    if "Churn" in df.columns:
        y_true = df["Churn"]
        X = df.drop(columns=["Churn", "CustomerID"])
    else:
        y_true = None
        X = df.drop(columns=["CustomerID"])

    log_pipeline_stage(_logger, "DRIFT CHECK (Incoming vs Training Reference)")
    reference_path = os.path.join(os.path.dirname(__file__), "..", "models", "reference_baseline.csv")
    if os.path.exists(reference_path):
        reference_df = pd.read_csv(reference_path)
        X_reference = reference_df.drop(columns=["CustomerID", "Churn"], errors="ignore")
        drift_results = run_drift_check(X_reference, X, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        summarize_drift(drift_results, DRIFT_PSI_WARNING, DRIFT_PSI_CRITICAL, 0.05, _logger)
    else:
        _logger.info("No reference baseline found — skipping drift check.")

    log_pipeline_stage(_logger, "PREDICT")
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    results = X.copy()
    results["Churn_Prediction"] = y_pred
    results["Churn_Probability"] = y_proba
    if y_true is not None:
        results["Churn_Actual"] = y_true.values

    _logger.info(f"Predictions complete — {len(results)} rows.")
    log_metrics(_logger, {"rows": len(results), "positive_rate": y_pred.mean():.4f}, stage="PREDICT")

    if output_path:
        results.to_csv(output_path, index=False)
        _logger.info(f"Results written to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run churn predictions from a saved model.")
    parser.add_argument("--model", required=True, help="Path to .joblib model file")
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--output", default=None, help="Output CSV path (optional)")
    args = parser.parse_args()

    predict(args.model, args.data, args.output)


if __name__ == "__main__":
    main()