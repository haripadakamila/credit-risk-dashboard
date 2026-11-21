"""
Phase 2B-1: Random Forest Classifier Training and Hyperparameter Tuning.

Methodology:
- 5-Fold Stratified Cross-Validation on the training set.
- Hyperparameter tuning via RandomizedSearchCV over depth, split, leaf, and feature sub-sampling.
- Rigorous comparison between class imbalance mitigation strategies:
    1. class_weight="balanced"
    2. SMOTE (strictly embedded inside an imblearn.pipeline.Pipeline to guarantee zero data leakage).
- Selects the superior variant based on CV ROC-AUC / PR-AUC.
- Refits the optimal configuration on the full training partition.
- Evaluates on the held-out untouched test set:
    Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Brier Score.
- Saves the trained model to models/random_forest.joblib.
- Appends or updates the Random Forest metrics in reports/model_comparison.csv
  without overwriting existing baseline entries.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
COMPARISON_CSV_PATH = REPORTS_DIR / "model_comparison.csv"
RANDOM_STATE = 42


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load preprocessed feature matrices and targets."""
    x_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    x_test_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    y_test_path = PROCESSED_DATA_DIR / "y_test.csv"

    for path in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing processed data file: {path}")

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

    print(f"[Data Loader] Loaded X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"[Data Loader] Target balance: Train 1s = {(y_train == 1).mean():.2%}, Test 1s = {(y_test == 1).mean():.2%}")
    return X_train, X_test, y_train, y_test


def tune_and_compare_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> Tuple[str, Any, Dict[str, float], Dict[str, Any]]:
    """
    Tune Random Forest with RandomizedSearchCV for both class_weight='balanced' and SMOTE.
    
    Compares both tuned models using full CV scoring metrics and returns the best.
    """
    print("\n" + "=" * 75)
    print("HYPERPARAMETER TUNING & VARIANT COMPARISON (5-FOLD STRATIFIED CV)")
    print("=" * 75)

    scoring = ["roc_auc", "average_precision", "f1", "recall"]

    # 1. Variant 1: class_weight='balanced'
    print("[Tuning] Variant 1: class_weight='balanced'...")
    param_dist_cw = {
        "n_estimators": [100, 150, 200],
        "max_depth": [12, 16, 20, 24],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [2, 4, 8],
        "max_features": ["sqrt", 0.6, 0.8],
    }
    rf_cw_base = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    search_cw = RandomizedSearchCV(
        rf_cw_base,
        param_distributions=param_dist_cw,
        n_iter=8,
        cv=cv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=1,  # Each RF uses n_jobs=-1 for optimal multithreading without thrashing
    )
    search_cw.fit(X_train, y_train)
    best_rf_cw = search_cw.best_estimator_
    best_params_cw = search_cw.best_params_

    # Comprehensive CV evaluation of best CW estimator
    cv_res_cw = cross_validate(best_rf_cw, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    cw_metrics = {
        "cv_roc_auc_mean": float(np.mean(cv_res_cw["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cv_res_cw["test_roc_auc"])),
        "cv_pr_auc_mean": float(np.mean(cv_res_cw["test_average_precision"])),
        "cv_pr_auc_std": float(np.std(cv_res_cw["test_average_precision"])),
        "cv_f1_mean": float(np.mean(cv_res_cw["test_f1"])),
        "cv_recall_mean": float(np.mean(cv_res_cw["test_recall"])),
    }
    print(f"  Best params (CW): {best_params_cw}")
    print(
        f"  [class_weight='balanced'] CV ROC-AUC: {cw_metrics['cv_roc_auc_mean']:.4f} (+/- {cw_metrics['cv_roc_auc_std']:.4f}) | "
        f"PR-AUC: {cw_metrics['cv_pr_auc_mean']:.4f} | F1: {cw_metrics['cv_f1_mean']:.4f}"
    )

    # 2. Variant 2: SMOTE Pipeline
    print("\n[Tuning] Variant 2: SMOTE Pipeline...")
    param_dist_smote = {
        "rf__n_estimators": [100, 150, 200],
        "rf__max_depth": [12, 16, 20, 24],
        "rf__min_samples_split": [5, 10, 15],
        "rf__min_samples_leaf": [2, 4, 8],
        "rf__max_features": ["sqrt", 0.6, 0.8],
    }
    pipe_smote_base = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    search_smote = RandomizedSearchCV(
        pipe_smote_base,
        param_distributions=param_dist_smote,
        n_iter=8,
        cv=cv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    search_smote.fit(X_train, y_train)
    best_pipe_smote = search_smote.best_estimator_
    best_params_smote = search_smote.best_params_

    # Comprehensive CV evaluation of best SMOTE estimator
    cv_res_smote = cross_validate(best_pipe_smote, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    smote_metrics = {
        "cv_roc_auc_mean": float(np.mean(cv_res_smote["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cv_res_smote["test_roc_auc"])),
        "cv_pr_auc_mean": float(np.mean(cv_res_smote["test_average_precision"])),
        "cv_pr_auc_std": float(np.std(cv_res_smote["test_average_precision"])),
        "cv_f1_mean": float(np.mean(cv_res_smote["test_f1"])),
        "cv_recall_mean": float(np.mean(cv_res_smote["test_recall"])),
    }
    print(f"  Best params (SMOTE): {best_params_smote}")
    print(
        f"  [SMOTE Pipeline] CV ROC-AUC: {smote_metrics['cv_roc_auc_mean']:.4f} (+/- {smote_metrics['cv_roc_auc_std']:.4f}) | "
        f"PR-AUC: {smote_metrics['cv_pr_auc_mean']:.4f} | F1: {smote_metrics['cv_f1_mean']:.4f}"
    )

    # 3. Decision rule: Select based on CV ROC-AUC, tie-breaker PR-AUC
    if cw_metrics["cv_roc_auc_mean"] >= smote_metrics["cv_roc_auc_mean"]:
        best_variant = 'class_weight="balanced"'
        best_model = best_rf_cw
        best_cv_metrics = cw_metrics
        best_params = best_params_cw
    else:
        best_variant = "SMOTE"
        best_model = best_pipe_smote
        best_cv_metrics = smote_metrics
        best_params = best_params_smote

    print(f"\n--> Selected Superior Random Forest Variant: '{best_variant}'")
    return best_variant, best_model, best_cv_metrics, best_params


def evaluate_on_test_set(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Compute comprehensive performance metrics on the held-out test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred)),
        "F1": float(f1_score(y_test, y_pred)),
        "ROC_AUC": float(roc_auc_score(y_test, y_prob)),
        "PR_AUC": float(average_precision_score(y_test, y_prob)),
        "Brier_Score": float(brier_score_loss(y_test, y_prob)),
    }
    return metrics


def append_to_comparison_csv(rf_record: Dict[str, Any]) -> pd.DataFrame:
    """
    Append or update the Random Forest record in reports/model_comparison.csv
    without overwriting existing rows.
    """
    if COMPARISON_CSV_PATH.exists():
        df_comp = pd.read_csv(COMPARISON_CSV_PATH)
        # Remove any previous Random Forest entry if re-running
        df_comp = df_comp[df_comp["Model"] != "Random Forest"]
        df_updated = pd.concat([df_comp, pd.DataFrame([rf_record])], ignore_index=True)
    else:
        df_updated = pd.DataFrame([rf_record])

    df_updated.to_csv(COMPARISON_CSV_PATH, index=False)
    print(f"\n[Artifact] Updated comparison metrics saved to: {COMPARISON_CSV_PATH}")
    return df_updated


def run_random_forest() -> pd.DataFrame:
    """End-to-end execution of Random Forest tuning, testing, and logging."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("PHASE 2B-1: RANDOM FOREST TRAINING & EVALUATION")
    print("=" * 75)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 2. Hyperparameter tuning & variant comparison
    best_variant, best_model, cv_metrics, best_params = tune_and_compare_rf(
        X_train=X_train,
        y_train=y_train,
        cv=cv,
    )

    # 3. Refit on full training set
    print(f"\nRefitting optimal Random Forest ({best_variant}) on full training partition...")
    best_model.fit(X_train, y_train)

    # 4. Evaluate on test set
    test_metrics = evaluate_on_test_set(best_model, X_test, y_test)
    print("\n" + "=" * 75)
    print("TEST SET EVALUATION RESULTS (RANDOM FOREST)")
    print("=" * 75)
    for metric_name, val in test_metrics.items():
        print(f"  {metric_name:<15}: {val:.4f}")

    # 5. Save model artifact
    rf_model_path = MODELS_DIR / "random_forest.joblib"
    joblib.dump(best_model, rf_model_path)
    print(f"\n[Artifact] Saved tuned Random Forest model to: {rf_model_path}")

    # 6. Append to model_comparison.csv
    rf_record = {
        "Model": "Random Forest",
        "Best_Variant": best_variant,
        "CV_ROC_AUC": cv_metrics["cv_roc_auc_mean"],
        "CV_PR_AUC": cv_metrics["cv_pr_auc_mean"],
        "CV_F1": cv_metrics["cv_f1_mean"],
        "Test_Accuracy": test_metrics["Accuracy"],
        "Test_Precision": test_metrics["Precision"],
        "Test_Recall": test_metrics["Recall"],
        "Test_F1": test_metrics["F1"],
        "Test_ROC_AUC": test_metrics["ROC_AUC"],
        "Test_PR_AUC": test_metrics["PR_AUC"],
        "Test_Brier_Score": test_metrics["Brier_Score"],
    }
    df_comparison = append_to_comparison_csv(rf_record)

    # 7. Print consolidated comparison table
    print("\n" + "=" * 105)
    print("CONSOLIDATED 3-MODEL COMPARISON TABLE (HELD-OUT TEST SET)")
    print("=" * 105)
    display_df = df_comparison[[
        "Model",
        "Best_Variant",
        "Test_Accuracy",
        "Test_Precision",
        "Test_Recall",
        "Test_F1",
        "Test_ROC_AUC",
        "Test_PR_AUC",
        "Test_Brier_Score",
    ]].copy()
    for col in display_df.select_dtypes(include="float").columns:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    print(display_df.to_string(index=False))
    print("=" * 105)

    return df_comparison


if __name__ == "__main__":
    run_random_forest()
