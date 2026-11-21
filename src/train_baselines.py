"""
Phase 2A: Baseline Classical Machine Learning Models.

Trains and evaluates:
1. Logistic Regression
2. Decision Tree

Methodology:
- Stratified 5-Fold Cross-Validation on the training set.
- Evaluates class imbalance mitigation strategies: class_weight='balanced' vs. SMOTE
  (SMOTE is strictly applied inside an imblearn Pipeline on training folds to prevent data leakage).
- Selects the superior variant based on 5-fold CV ROC-AUC / PR-AUC.
- Refits the selected variant on the full training set.
- Evaluates performance on the held-out untouched test set:
  Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Brier Score.
- Saves fitted models to models/ and consolidated metrics to reports/model_comparison.csv.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
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


def evaluate_imbalance_variants(
    model_name: str,
    variants: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> Tuple[str, Any, Dict[str, float]]:
    """
    Perform 5-fold Stratified Cross-Validation on each imbalance strategy.
    
    Returns the best variant name, the un-fitted best estimator/pipeline, and CV scores.
    """
    scoring = ["roc_auc", "average_precision", "f1", "recall"]
    results = {}

    print(f"\n--- 5-Fold Stratified CV for {model_name} ---")
    for variant_name, estimator in variants.items():
        cv_res = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        mean_roc = float(np.mean(cv_res["test_roc_auc"]))
        std_roc = float(np.std(cv_res["test_roc_auc"]))
        mean_pr = float(np.mean(cv_res["test_average_precision"]))
        std_pr = float(np.std(cv_res["test_average_precision"]))
        mean_f1 = float(np.mean(cv_res["test_f1"]))
        mean_recall = float(np.mean(cv_res["test_recall"]))

        results[variant_name] = {
            "cv_roc_auc_mean": mean_roc,
            "cv_roc_auc_std": std_roc,
            "cv_pr_auc_mean": mean_pr,
            "cv_pr_auc_std": std_pr,
            "cv_f1_mean": mean_f1,
            "cv_recall_mean": mean_recall,
            "estimator": estimator,
        }
        print(
            f"  [{variant_name}] ROC-AUC: {mean_roc:.4f} (+/- {std_roc:.4f}) | "
            f"PR-AUC: {mean_pr:.4f} (+/- {std_pr:.4f}) | "
            f"F1: {mean_f1:.4f} | Recall: {mean_recall:.4f}"
        )

    # Primary selection criteria: ROC-AUC, breaking ties with PR-AUC
    best_variant_name = max(
        results.keys(),
        key=lambda k: (results[k]["cv_roc_auc_mean"], results[k]["cv_pr_auc_mean"]),
    )
    best_info = results[best_variant_name]
    print(f"  --> Selected Superior Variant: '{best_variant_name}'")
    return best_variant_name, best_info["estimator"], best_info


def evaluate_on_test_set(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics on the held-out test set."""
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


def run_baselines() -> pd.DataFrame:
    """Execute training, cross-validation, selection, testing, and serialization."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("PHASE 2A: BASELINE CLASSICAL ML MODELS TRAINING & EVALUATION")
    print("=" * 75)

    X_train, X_test, y_train, y_test = load_processed_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1. Define model configurations and variants
    models_config = {
        "Logistic Regression": {
            "save_path": MODELS_DIR / "logistic_regression.joblib",
            "variants": {
                'class_weight="balanced"': LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
                "SMOTE": ImbPipeline([
                    ("smote", SMOTE(random_state=RANDOM_STATE)),
                    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
                ]),
            },
        },
        "Decision Tree": {
            "save_path": MODELS_DIR / "decision_tree.joblib",
            "variants": {
                'class_weight="balanced"': DecisionTreeClassifier(
                    class_weight="balanced",
                    max_depth=8,
                    min_samples_leaf=20,
                    random_state=RANDOM_STATE,
                ),
                "SMOTE": ImbPipeline([
                    ("smote", SMOTE(random_state=RANDOM_STATE)),
                    ("clf", DecisionTreeClassifier(
                        max_depth=8,
                        min_samples_leaf=20,
                        random_state=RANDOM_STATE,
                    )),
                ]),
            },
        },
    }

    comparison_records: List[Dict[str, Any]] = []

    for model_name, cfg in models_config.items():
        # A. Cross-Validation and Variant Selection
        best_variant, best_model, cv_info = evaluate_imbalance_variants(
            model_name=model_name,
            variants=cfg["variants"],
            X_train=X_train,
            y_train=y_train,
            cv=cv,
        )

        # B. Refit selected variant on entire training set
        print(f"  Fitting best {model_name} ({best_variant}) on full training set...")
        best_model.fit(X_train, y_train)

        # C. Evaluate on untouched test set
        test_metrics = evaluate_on_test_set(best_model, X_test, y_test)
        print(f"  Test Evaluation for {model_name}:")
        print(
            f"    ROC-AUC: {test_metrics['ROC_AUC']:.4f} | PR-AUC: {test_metrics['PR_AUC']:.4f} | "
            f"F1: {test_metrics['F1']:.4f} | Brier Score: {test_metrics['Brier_Score']:.4f}"
        )

        # D. Save fitted model
        joblib.dump(best_model, cfg["save_path"])
        print(f"  Saved model artifact to: {cfg['save_path']}")

        # E. Record metrics
        record = {
            "Model": model_name,
            "Best_Variant": best_variant,
            "CV_ROC_AUC": cv_info["cv_roc_auc_mean"],
            "CV_PR_AUC": cv_info["cv_pr_auc_mean"],
            "CV_F1": cv_info["cv_f1_mean"],
            "Test_Accuracy": test_metrics["Accuracy"],
            "Test_Precision": test_metrics["Precision"],
            "Test_Recall": test_metrics["Recall"],
            "Test_F1": test_metrics["F1"],
            "Test_ROC_AUC": test_metrics["ROC_AUC"],
            "Test_PR_AUC": test_metrics["PR_AUC"],
            "Test_Brier_Score": test_metrics["Brier_Score"],
        }
        comparison_records.append(record)

    # 2. Save comparison table to reports/model_comparison.csv
    df_comparison = pd.DataFrame(comparison_records)
    csv_path = REPORTS_DIR / "model_comparison.csv"
    df_comparison.to_csv(csv_path, index=False)
    print(f"\n[Artifact] Saved model comparison metrics to: {csv_path}")

    # 3. Print final formatted summary table
    print("\n" + "=" * 90)
    print("BASELINE MODELS EVALUATION SUMMARY (HELD-OUT TEST SET)")
    print("=" * 90)
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
    print("=" * 90)

    return df_comparison


if __name__ == "__main__":
    run_baselines()
