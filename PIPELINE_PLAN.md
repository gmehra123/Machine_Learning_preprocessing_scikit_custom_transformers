# Machine Learning Pipeline Plan

## Overview

A production-ready sklearn pipeline for customer churn prediction, built as modular `.py` scripts with drift detection.

---

## Architecture

```
project/
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py              # hyperparameters, column names, thresholds
│   │   └── logging.py             # structured logger
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py             # data loading + train/test split
│   │   └── drift.py               # PSI + chi-squared drift detection
│   ├── features/
│   │   ├── __init__.py
│   │   ├── transformers.py        # GroupedImputerFull, OutlierClipper
│   │   ├── encoders.py            # Label/OneHot encoder wrappers
│   │   └── scalers.py             # Standard/MinMax scaler wrappers
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── builder.py             # constructs sklearn Pipeline in code
│   └── models/
│       ├── __init__.py
│       ├── train.py               # training loop with metrics logging
│       └── registry.py            # model save/load (joblib)
├── scripts/
│   ├── train.py                   # main training entry point
│   └── predict.py                 # prediction entry point
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py           # smoke tests
└── models/                        # created at runtime
```

---

## Implementation Order

### Phase 1 — Foundation

**1. `src/utils/config.py`**
- `TARGET_COL = "Churn"`
- Numeric features: `["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend", "Last Interaction"]`
- Categorical features: `["Gender", "Subscription Type", "Contract Length"]`
- ID column: `"CustomerID"` (dropped before modeling)
- Imputer group column: `"Gender"`
- Train/test split: `test_size=0.15`, `random_state=42`, stratify on target
- Drift thresholds: PSI `> 0.2` warning, `> 0.25` critical; chi-squared `p < 0.05` drift
- Model hyperparameters dict

**2. `src/utils/logging.py`**
- `get_logger(name)` → structured Python logger
- Methods: `log_metrics()`, `log_drift_warning()`, `log_pipeline_stage()`

---

### Phase 2 — Data Layer

**3. `src/data/loaders.py`**
- `load_csv(path)` → DataFrame
- `split_data(df, target_col, test_size, random_state)` → `(X_train, X_test, y_train, y_test)` with stratify

**4. `src/data/drift.py`**
- `compute_psi(expected, actual, buckets=10)` → float PSI score
  - Quantile-based bucketing, expected vs actual proportions per bucket
- `detect_numeric_drift(train_series, new_series, threshold=0.2)` → `{"psi": float, "drifted": bool}`
- `detect_categorical_drift(train_series, new_series, threshold=0.05)` → `{"chi2": float, "p_value": float, "drifted": bool}`
- `run_drift_check(X_train, X_new, numeric_cols, categorical_cols)` → dict of per-feature drift results
- `summarize_drift(drift_results)` → human-readable summary + warn/critical raise

---

### Phase 3 — Feature Engineering

**5. `src/features/transformers.py`**
- `GroupedImputerFull` — imputes all NaN columns using group mean, global mean fallback; implements `BaseEstimator, TransformerMixin`
- `OutlierClipper` — clips numeric columns at quantile-based bounds; fitted on train, transforms test

**6. `src/features/encoders.py`**
- `ColumnLabelEncoder` — wraps `LabelEncoder` for specified columns
- `ColumnOneHotEncoder` — wraps `OneHotEncoder` for specified columns

**7. `src/features/scalers.py`**
- `ColumnStandardScaler` — wraps `StandardScaler` for specified columns
- `ColumnMinMaxScaler` — wraps `MinMaxScaler` for specified columns

---

### Phase 4 — Pipeline Construction

**8. `src/pipeline/builder.py`**
- `build_pipeline(config)` → `sklearn.Pipeline`
  - Steps:
    1. Imputation → `GroupedImputerFull`
    2. Outlier clipping → `OutlierClipper`
    3. Encoding → `ColumnOneHotEncoder`
    4. Scaling → `ColumnStandardScaler`
    5. Model → instantiated from config

---

### Phase 5 — Model Training & Registry

**9. `src/models/registry.py`**
- `save_model(pipeline, path)` — `joblib.dump`
- `load_model(path)` — `joblib.load`
- `get_model_path(name, version)` → path under `./models/`

**10. `src/models/train.py`**
- `train_pipeline(X_train, y_train, config)` → fitted Pipeline
  - Run drift check before training
  - Log: accuracy, precision, recall, F1, ROC-AUC on train/validation split
- `evaluate_pipeline(pipeline, X_test, y_test)` → dict of metrics

---

### Phase 6 — Entry Points

**11. `scripts/train.py`**
- Load training data
- Run drift detection (baseline vs full dataset)
- Split train/test
- Build pipeline via `builder.build_pipeline()`
- Train and evaluate
- Save model to `./models/churn_model_v{version}.joblib`
- Print final metrics + drift summary

**12. `scripts/predict.py`**
- Load saved model
- Load test data
- Apply drift detection on incoming data vs training baseline
- Output predictions + probabilities
- `--output` flag to write results to CSV

---

### Phase 7 — Testing

**13. `tests/test_pipeline.py`**
- `test_imputer_fills_nans()`
- `test_clipper_bounds()`
- `test_pipeline_shape()`
- `test_psi_calculation()`
- `test_model_save_load()`
- `test_drift_detection_triggers()`
- `test_train_pipeline_no_crash()`

---

## Dependencies

```
pandas
scikit-learn
numpy
joblib
pytest
```