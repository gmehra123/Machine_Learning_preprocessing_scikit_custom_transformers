"""Column-wise scaler wrappers for sklearn."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler, MinMaxScaler as SklearnMinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnStandardScaler(BaseEstimator, TransformerMixin):
    """Wraps sklearn StandardScaler for application to specified columns."""

    def __init__(self, columns: list):
        self.columns = columns
        self.scaler = SklearnStandardScaler()

    def fit(self, X, y=None):
        df = self._get_df(X)
        self.scaler.fit(df[self.columns])
        return self

    def transform(self, X, y=None):
        df = self._get_df(X).copy()
        df[self.columns] = self.scaler.transform(df[self.columns])
        return df

    def _get_df(self, X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


class ColumnMinMaxScaler(BaseEstimator, TransformerMixin):
    """Wraps sklearn MinMaxScaler for application to specified columns."""

    def __init__(self, columns: list):
        self.columns = columns
        self.scaler = SklearnMinMaxScaler()

    def fit(self, X, y=None):
        df = self._get_df(X)
        self.scaler.fit(df[self.columns])
        return self

    def transform(self, X, y=None):
        df = self._get_df(X).copy()
        df[self.columns] = self.scaler.transform(df[self.columns])
        return df

    def _get_df(self, X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)