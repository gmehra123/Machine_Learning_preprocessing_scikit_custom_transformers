"""Model training loop with drift-aware validation."""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.pipeline.builder import build_pipeline
from src.data.drift import run_drift_check, summarize_drift
from src.utils.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, DRIFT_PSI_WARNING, DRIFT_PSI_CRITICAL
from src.utils.logging import get_logger, log_metrics, log_pipeline_stage


_logger = get_logger(__name__)


def train_pipeline(X_train, y_train, config: dict = None) -> object:
    """Train the pipeline with drift detection and metrics logging.

    Parameters
    ----------
    X_train : DataFrame
        Training features.
    y_train : Series
        Training target.
    config : dict, optional
        Override config dict. If None, uses src.utils.config defaults.

    Returns
    -------
    fitted_pipeline : Pipeline
    """
    config = config or {}

    log_pipeline_stage(_logger, "DRIFT CHECK (Training vs Baseline)")
    drift_results = run_drift_check(
        X_train, X_train,
        numeric_cols=NUMERIC_FEATURES,
        categorical_cols=CATEGORICAL_FEATURES,
    )
    has_drift = summarize_drift(
        drift_results,
        psi_warning=DRIFT_PSI_WARNING,
        psi_critical=DRIFT_PSI_CRITICAL,
        chi2_threshold=0.05,
        logger=_logger,
    )
    if has_drift:
        _logger.warning("Drift detected in training data — proceeding with caution.")

    log_pipeline_stage(_logger, "TRAIN SPLIT (Train / Val)")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=config.get("val_size", 0.15),
        stratify=y_train,
        random_state=config.get("random_state", 42),
    )

    log_pipeline_stage(_logger, "BUILD PIPELINE")
    pipeline = build_pipeline(
        numeric_features=config.get("numeric_features", None),
        categorical_features=config.get("categorical_features", None),
        group_col=config.get("group_col", None),
        model_type=config.get("model_type", None),
        model_params=config.get("model_params", None),
        outlier_lower_quantile=config.get("outlier_lower_quantile", 0.01),
        outlier_upper_quantile=config.get("outlier_upper_quantile", 0.99),
    )

    log_pipeline_stage(_logger, "FIT")
    pipeline.fit(X_tr, y_tr)

    log_pipeline_stage(_logger, "EVALUATE ON VALIDATION SET")
    y_val_pred = pipeline.predict(X_val)
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]

    val_metrics = {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "precision": precision_score(y_val, y_val_pred),
        "recall": recall_score(y_val, y_val_pred),
        "f1": f1_score(y_val, y_val_pred),
        "roc_auc": roc_auc_score(y_val, y_val_proba),
    }
    log_metrics(_logger, val_metrics, stage="VAL")
    return pipeline


def evaluate_pipeline(pipeline, X_test, y_test) -> dict:
    """Evaluate a fitted pipeline on held-out test data.

    Returns
    -------
    dict of metrics
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics