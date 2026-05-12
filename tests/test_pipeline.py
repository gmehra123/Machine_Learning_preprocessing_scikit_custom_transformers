"""Pipeline smoke tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from src.data.loaders import load_csv, split_data
from src.features.transformers import GroupedImputerFull, OutlierClipper
from src.pipeline.builder import build_pipeline
from src.data.drift import compute_psi, detect_numeric_drift, detect_categorical_drift
from src.models.registry import save_model, load_model
from src.utils.config import TARGET_COL


def test_imputer_fills_nans():
    df = pd.DataFrame({
        "Gender": ["M", "F", "M", "F"],
        "A": [1.0, np.nan, 3.0, np.nan],
        "B": [np.nan, 2.0, np.nan, 4.0],
    })
    imputer = GroupedImputerFull(group_col="Gender")
    result = imputer.fit_transform(df)
    assert result.isna().sum().sum() == 0, "Imputer should fill all NaNs"


def test_clipper_bounds():
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 100.0, 200.0]})
    clipper = OutlierClipper(columns=["A"], lower_quantile=0.0, upper_quantile=0.95)
    clipper.fit(df)
    result = clipper.transform(df)
    upper_bound = df["A"].quantile(0.95)
    assert result["A"].max() <= upper_bound, "Values should be clipped to upper quantile"


def test_pipeline_shape():
    df = load_csv("customer_churn_dataset-training-master.csv")
    df.dropna(subset=[TARGET_COL], inplace=True)
    X_train, X_test, y_train, y_test = split_data(df, TARGET_COL, 0.15, 42)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    assert len(preds) == len(y_test), "Predictions should match test set length"


def test_psi_calculation():
    a = np.random.normal(50, 10, 1000)
    b = np.random.normal(55, 10, 1000)
    psi = compute_psi(a, b)
    assert 0 <= psi < 10, f"PSI should be non-negative, got {psi}"
    assert psi > 0, "Different distributions should produce positive PSI"


def test_model_save_load():
    df = load_csv("customer_churn_dataset-training-master.csv")
    df.dropna(subset=[TARGET_COL], inplace=True)
    X, _, y, _ = split_data(df, TARGET_COL, 0.15, 42)
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    path = save_model(pipeline, name="test_model", version=999)
    loaded = load_model(path)
    preds_original = pipeline.predict(X)
    preds_loaded = loaded.predict(X)
    assert np.array_equal(preds_original, preds_loaded), "Loaded model should produce same predictions"
    os.remove(path)


def test_drift_detection_triggers():
    train = pd.Series(np.random.normal(50, 10, 1000))
    drifted = pd.Series(np.random.normal(70, 10, 1000))
    result = detect_numeric_drift(train, drifted, threshold=0.2)
    assert result["drifted"] == True, "Large shift should trigger drift detection"


def test_train_pipeline_no_crash():
    df = load_csv("customer_churn_dataset-training-master.csv")
    df.dropna(subset=[TARGET_COL], inplace=True)
    X, _, y, _ = split_data(df, TARGET_COL, 0.85, 42)
    pipeline = build_pipeline()
    fitted = pipeline.fit(X.iloc[:1000], y.iloc[:1000])
    assert fitted is not None, "Pipeline should fit without crashing on small sample"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])