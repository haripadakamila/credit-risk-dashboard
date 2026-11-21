"""
Leakage-Free Preprocessing Pipeline for Credit Risk Prediction.

This module performs:
1. Integrity cleaning: Removal of exact duplicates and physical impossibilities
   (person_age > 100, person_emp_length > 60).
2. Stratified 80/20 train-test split (random_state=42) FIRST to guarantee zero data leakage.
3. Imputation of missing values (person_emp_length, loan_int_rate) strictly using training-set medians.
4. Domain feature engineering:
   - loan_to_income_ratio: loan_amnt / person_income
   - interest_to_income_burden: (loan_amnt * (loan_int_rate / 100)) / person_income
   - credit_history_to_age_ratio: cb_person_cred_hist_length / person_age
5. Scaling of numerical columns and one-hot encoding of categorical columns
   (fitted strictly on the training set).
6. Export of processed datasets to data/processed/ and fitted preprocessor to models/preprocessor.joblib.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure root directory is in sys.path for clean module imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RAW_DATA_PATH = Path("data/raw/credit_risk_dataset.csv")
PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

# Feature definitions
BASE_NUMERICAL_COLS: List[str] = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

ENGINEERED_NUMERICAL_COLS: List[str] = [
    "loan_to_income_ratio",
    "interest_to_income_burden",
    "credit_history_to_age_ratio",
]

CATEGORICAL_COLS: List[str] = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

TARGET_COL: str = "loan_status"


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate records and unrealistic outliers prior to splitting.
    
    Outlier boundaries:
    - person_age > 100
    - person_emp_length > 60
    """
    initial_count = len(df)
    
    # 1. Remove duplicate records
    df_cleaned = df.drop_duplicates()
    dups_removed = initial_count - len(df_cleaned)
    
    # 2. Filter physical impossibilities
    valid_age_mask = df_cleaned["person_age"] <= 100
    valid_emp_mask = df_cleaned["person_emp_length"].isna() | (df_cleaned["person_emp_length"] <= 60)
    df_cleaned = df_cleaned[valid_age_mask & valid_emp_mask]
    outliers_removed = initial_count - dups_removed - len(df_cleaned)
    
    print(f"[Data Cleaning] Removed {dups_removed} duplicates and {outliers_removed} extreme outlier records.")
    print(f"[Data Cleaning] Dataset size after cleaning: {len(df_cleaned):,} rows.")
    return df_cleaned


class CreditRiskPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leakage-free credit risk preprocessor.
    
    Learns medians, scaling statistics, and one-hot categories exclusively
    on the training set and transforms both training and unseen inference data.
    """

    def __init__(self) -> None:
        self.base_num_cols: List[str] = BASE_NUMERICAL_COLS
        self.eng_num_cols: List[str] = ENGINEERED_NUMERICAL_COLS
        self.cat_cols: List[str] = CATEGORICAL_COLS
        
        # Stored during fit
        self.medians_: Dict[str, float] = {}
        self.scaler_: StandardScaler = StandardScaler()
        self.encoder_: OneHotEncoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.all_num_cols_: List[str] = []
        self.encoded_cat_cols_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.is_fitted_: bool = False

    def _engineer_features(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values and compute domain features."""
        df_out = X_df.copy()
        
        # 1. Impute missing values using learned training medians
        for col, median_val in self.medians_.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(median_val)
        
        # 2. Calculate domain engineered features
        # Ratio of requested loan amount to annual income
        df_out["loan_to_income_ratio"] = df_out["loan_amnt"] / df_out["person_income"]
        
        # Annual interest obligation burden relative to annual income
        df_out["interest_to_income_burden"] = (
            df_out["loan_amnt"] * (df_out["loan_int_rate"] / 100.0)
        ) / df_out["person_income"]
        
        # Ratio of credit history length to applicant age
        df_out["credit_history_to_age_ratio"] = (
            df_out["cb_person_cred_hist_length"] / df_out["person_age"]
        )
        
        return df_out

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CreditRiskPreprocessor":
        """
        Fit preprocessor strictly on training data.
        
        Computes training medians, fits StandardScaler and OneHotEncoder.
        """
        X_df = X.copy()
        
        # 1. Learn training medians for imputation
        self.medians_ = {
            "person_emp_length": float(X_df["person_emp_length"].median()),
            "loan_int_rate": float(X_df["loan_int_rate"].median()),
        }
        
        # 2. Engineer domain features on imputed training data
        X_eng = self._engineer_features(X_df)
        
        # 3. Fit StandardScaler on all numerical columns (base + engineered)
        self.all_num_cols_ = self.base_num_cols + self.eng_num_cols
        self.scaler_.fit(X_eng[self.all_num_cols_])
        
        # 4. Fit OneHotEncoder on categorical columns
        self.encoder_.fit(X_eng[self.cat_cols])
        self.encoded_cat_cols_ = list(self.encoder_.get_feature_names_out(self.cat_cols))
        
        # 5. Composite feature names
        self.feature_names_out_ = self.all_num_cols_ + self.encoded_cat_cols_
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new or test data using fitted transformers.
        
        Returns a formatted DataFrame with 29 named features.
        """
        if not self.is_fitted_:
            raise RuntimeError("CreditRiskPreprocessor must be fitted before calling transform.")
        
        X_df = X.copy()
        
        # Impute and engineer features
        X_eng = self._engineer_features(X_df)
        
        # Scale numericals
        num_scaled = self.scaler_.transform(X_eng[self.all_num_cols_])
        
        # Encode categoricals
        cat_encoded = self.encoder_.transform(X_eng[self.cat_cols])
        
        # Combine into DataFrame
        transformed_matrix = np.hstack([num_scaled, cat_encoded])
        return pd.DataFrame(
            transformed_matrix,
            columns=self.feature_names_out_,
            index=X_df.index,
        )

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Return names of output features."""
        return self.feature_names_out_


# Explicitly assign module for consistent pickling and unpickling across packages
CreditRiskPreprocessor.__module__ = "src.preprocess"


def run_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Execute end-to-end preprocessing, validation, and export."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ensure src.preprocess is registered in sys.modules
    if "src.preprocess" not in sys.modules:
        import src.preprocess
        sys.modules["src.preprocess"].CreditRiskPreprocessor = CreditRiskPreprocessor
    else:
        sys.modules["src.preprocess"].CreditRiskPreprocessor = CreditRiskPreprocessor
    
    print("=" * 70)
    print("Executing Leakage-Free Preprocessing Pipeline...")
    print("=" * 70)
    
    # 1. Load raw dataset
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH}")
    
    df_raw = pd.read_csv(RAW_DATA_PATH)
    print(f"[Raw Data] Initial shape: {df_raw.shape}")
    
    # 2. Integrity cleaning
    df_clean = clean_raw_data(df_raw)
    
    # 3. Separate features and target
    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]
    
    # 4. Stratified 80/20 train-test split FIRST to guarantee zero leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    
    print(f"\n[Split] Train split size: {len(X_train):,} ({len(X_train)/len(X):.1%})")
    print(f"[Split] Test split size:  {len(X_test):,} ({len(X_test)/len(X):.1%})")
    
    # 5. Fit preprocessor strictly on training set
    preprocessor = CreditRiskPreprocessor()
    preprocessor.fit(X_train)
    print(f"[Preprocessor] Learned Training Medians: {preprocessor.medians_}")
    
    # 6. Transform both train and test partitions
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 7. Verification checks
    assert X_train_processed.isna().sum().sum() == 0, "Missing values found in X_train_processed!"
    assert X_test_processed.isna().sum().sum() == 0, "Missing values found in X_test_processed!"
    assert not np.isinf(X_train_processed.values).any(), "Infinite values found in X_train_processed!"
    assert not np.isinf(X_test_processed.values).any(), "Infinite values found in X_test_processed!"
    assert len(preprocessor.feature_names_out_) == 29, f"Expected 29 features, got {len(preprocessor.feature_names_out_)}"
    
    # 8. Create full consolidated datasets with target
    train_full = X_train_processed.copy()
    train_full[TARGET_COL] = y_train.values
    
    test_full = X_test_processed.copy()
    test_full[TARGET_COL] = y_test.values
    
    # 9. Save artifacts to data/processed/ and models/
    train_full_path = PROCESSED_DATA_DIR / "train.csv"
    test_full_path = PROCESSED_DATA_DIR / "test.csv"
    X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    X_test_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    y_test_path = PROCESSED_DATA_DIR / "y_test.csv"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    
    train_full.to_csv(train_full_path, index=False)
    test_full.to_csv(test_full_path, index=False)
    X_train_processed.to_csv(X_train_path, index=False)
    X_test_processed.to_csv(X_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)
    joblib.dump(preprocessor, preprocessor_path)
    
    print(f"\n[Storage] Saved train dataset to: {train_full_path}")
    print(f"[Storage] Saved test dataset to:  {test_full_path}")
    print(f"[Storage] Saved preprocessor to:  {preprocessor_path}")
    
    # 10. Display comprehensive status summary
    print("\n" + "=" * 70)
    print("FINAL PREPROCESSING VERIFICATION & METRICS")
    print("=" * 70)
    print(f"X_train Shape:         {X_train_processed.shape}")
    print(f"X_test Shape:          {X_test_processed.shape}")
    print(f"Total Features:        {X_train_processed.shape[1]}")
    print("-" * 70)
    print("Target Class Distribution:")
    print(f"  Train: 0 (Non-Default) = {(y_train == 0).sum():,} ({(y_train == 0).mean():.2%}) | "
          f"1 (Default) = {(y_train == 1).sum():,} ({(y_train == 1).mean():.2%})")
    print(f"  Test:  0 (Non-Default) = {(y_test == 0).sum():,} ({(y_test == 0).mean():.2%}) | "
          f"1 (Default) = {(y_test == 1).sum():,} ({(y_test == 1).mean():.2%})")
    print("-" * 70)
    print("Missing Value Counts:")
    print(f"  X_train_processed NaNs: {X_train_processed.isna().sum().sum()}")
    print(f"  X_test_processed NaNs:  {X_test_processed.isna().sum().sum()}")
    print("-" * 70)
    print("Engineered & Processed Feature Names:")
    for i, name in enumerate(preprocessor.feature_names_out_, 1):
        print(f"  {i:2d}. {name}")
    print("=" * 70)
    
    return X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":
    run_pipeline()
