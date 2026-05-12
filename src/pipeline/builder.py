"""Builds sklearn Pipeline from configuration (code-driven, no YAML)."""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.features.transformers import GroupedImputerFull, OutlierClipper
from src.features.encoders import ColumnOneHotEncoder
from src.features.scalers import ColumnStandardScaler
from src.utils.config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, IMPUTER_GROUP_COL,
    MODEL_TYPE, MODEL_PARAMS
)


_MODEL_CLASSES = {
    "LogisticRegression": LogisticRegression,
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
}


def build_pipeline(
    numeric_features: list = None,
    categorical_features: list = None,
    group_col: str = None,
    model_type: str = None,
    model_params: dict = None,
    outlier_lower_quantile: float = 0.01,
    outlier_upper_quantile: float = 0.99,
) -> Pipeline:
    """Construct a sklearn Pipeline from code-driven configuration.

    Pipeline stages (in order):
        1. GroupedImputerFull   — handles NaN imputation via group means
        2. OutlierClipper       — clips numeric features to quantiles
        3. ColumnOneHotEncoder  — one-hot encodes categorical features
        4. ColumnStandardScaler — standardizes numeric features
        5. Model                — classifier (LogisticRegression / RandomForest / etc.)
    """
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    if group_col is None:
        group_col = IMPUTER_GROUP_COL
    if model_type is None:
        model_type = MODEL_TYPE
    if model_params is None:
        model_params = MODEL_PARAMS

    model_class = _MODEL_CLASSES.get(model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(_MODEL_CLASSES.keys())}")

    return Pipeline([
        ("imputer", GroupedImputerFull(group_col=group_col)),
        ("clipper", OutlierClipper(
            columns=numeric_features,
            lower_quantile=outlier_lower_quantile,
            upper_quantile=outlier_upper_quantile,
        )),
        ("encoder", ColumnOneHotEncoder(columns=categorical_features, drop_first=True)),
        ("scaler", ColumnStandardScaler(columns=numeric_features)),
        ("model", model_class(**model_params)),
    ])


def get_pipeline_with_stages(pipeline: Pipeline) -> dict:
    """Return a dict of stage name -> fitted transformer for inspection."""
    return {name: step for name, step in pipeline.steps}