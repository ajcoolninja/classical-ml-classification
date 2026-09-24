import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

class PdaysImputer(BaseEstimator, TransformerMixin):
    """Custom imputer for the 'pdays' column."""
    def __init__(self):
        self.median_ = None
        
    def fit(self, X, y=None):
        pdays_series = X['pdays']
        self.median_ = pdays_series[pdays_series != -1].median()
        return self
        
    def transform(self, X):
        X_transformed = X.copy()
        # Create indicator for clients not previously contacted
        X_transformed['pdays_not_contacted'] = (X_transformed['pdays'] == -1).astype(int)
        # Impute -1 with the median of previously contacted clients
        X_transformed['pdays'] = X_transformed['pdays'].replace(-1, self.median_)
        return X_transformed[['pdays', 'pdays_not_contacted']]
        
    def get_feature_names_out(self, input_features=None):
        return np.array(['pdays', 'pdays_not_contacted'])


def get_preprocessors(X):
    """
    Builds and returns the scaled and unscaled ColumnTransformers.
    """
    # 1. Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
    
    # 2. Identify numeric columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 3. Exclude specific columns from standard numeric processing
    if 'duration' in numeric_cols:
        numeric_cols.remove('duration')
    if 'pdays' in numeric_cols:
        numeric_cols.remove('pdays')
        
    pdays_col = ['pdays']
    
    # 4. Build Scaled Preprocessor
    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), categorical_cols),
            ('pdays_custom', PdaysImputer(), pdays_col),
            ('num', RobustScaler(), numeric_cols)
        ],
        remainder='drop'
    )
    preprocessor_scaled.set_output(transform="pandas")
    
    # 5. Build Unscaled Preprocessor
    preprocessor_unscaled = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), categorical_cols),
            ('pdays_custom', PdaysImputer(), pdays_col),
            ('num', 'passthrough', numeric_cols)
        ],
        remainder='drop'
    )
    preprocessor_unscaled.set_output(transform="pandas")
    
    return preprocessor_scaled, preprocessor_unscaled