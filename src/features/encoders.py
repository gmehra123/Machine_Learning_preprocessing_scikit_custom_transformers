"""Column-wise encoder wrappers for sklearn."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder, OneHotEncoder as SklearnOneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        self.columns = columns
        self.encoders_ = {}

    def fit(self, X, y=None):
        df = self._get_df(X)
        for col in self.columns:
            le = SklearnLabelEncoder()
            le.fit(df[col].astype(str))
            self.encoders_[col] = le
        return self

    def transform(self, X, y=None):
        df = self._get_df(X).copy()
        for col in self.columns:
            df[col] = self.encoders_[col].transform(df[col].astype(str))
        return df

    def _get_df(self, X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


class ColumnOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list, drop_first: bool = True, sparse_output: bool = False):
        self.columns = columns
        self.drop_first = drop_first
        self.sparse_output = sparse_output
        drop_value = "first" if drop_first else None
        self.encoder_ = SklearnOneHotEncoder(drop=drop_value, sparse_output=sparse_output, handle_unknown="ignore")

    def fit(self, X, y=None):
        df = self._get_df(X)
        self.encoder_.fit(df[self.columns])
        self._feature_names = self.encoder_.get_feature_names_out(self.columns)
        return self

    def transform(self, X, y=None):
        df = self._get_df(X).copy()
        encoded = self.encoder_.transform(df[self.columns])
        encoded_df = pd.DataFrame(encoded, columns=self._feature_names, index=df.index)
        remaining_cols = [c for c in df.columns if c not in self.columns]
        result = pd.concat([df[remaining_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        return result

    def _get_df(self, X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
