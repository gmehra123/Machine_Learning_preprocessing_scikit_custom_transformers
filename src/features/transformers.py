"""Feature transformers: imputation and outlier clipping."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupedImputerFull(BaseEstimator, TransformerMixin):
    """Impute NaN values using group-wise mean, falling back to global mean."""

    def __init__(self, group_col: str):
        self.group_col = group_col

    def fit(self, X, y=None):
        self.na_cols_ = X.columns[X.isna().any()].tolist()
        self.grouped_means_ = X.groupby(self.group_col)[self.na_cols_].mean()
        self.global_means_ = X[self.na_cols_].mean()
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.na_cols_:
            fill_values = X[self.group_col].map(self.grouped_means_[col]).fillna(self.global_means_[col])
            X[col] = X[col].fillna(fill_values)
        return X


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip outliers to quantile-based bounds, fitted on training data."""

    def __init__(self, columns: list = None, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.columns = columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X = self._get_df(X)
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.lower_bounds_ = X[self.columns].quantile(self.lower_quantile)
        self.upper_bounds_ = X[self.columns].quantile(self.upper_quantile)
        return self

    def transform(self, X, y=None):
        X = self._get_df(X).copy()
        for col in self.columns:
            lower = self.lower_bounds_[col]
            upper = self.upper_bounds_[col]
            X[col] = X[col].clip(lower=lower, upper=upper)
        return X

    def _get_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)