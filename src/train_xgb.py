"""
Phase 2B-2: XGBoost Classifier Training, Hyperparameter Tuning, and Classical ML Wrap-Up.

Methodology:
1. 5-Fold Stratified Cross-Validation on the training partition.
2. Hyperparameter tuning via RandomizedSearchCV over depth, learning rate, subsample,
   colsample_bytree, and regularization.
3. Imbalance mitigation comparison: scale_pos_weight vs. SMOTE
   (SMOTE embedded inside an imblearn.pipeline.Pipeline to guarantee zero data leakage).
4. Full evaluation on held-out test set:
   Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, and Brier Score.
5. Model serialization:
   - models/xgboost.joblib
   - models/best_classical_model.joblib (top performing classical model)
6. Updates reports/model_comparison.csv preserving all previous models.
7. Visualizations generated and saved to reports/figures/:
   - classical_roc_curves.png (comparing all 4 classical models)
   - best_classical_confusion_matrix.png (for the top classical model)
"""

from pathlib import Path
from typing import Any, Dict, Tuple

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from xgboost import XGBClassifier

# Styling configurations for publication-grade visualizations
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#ebebeb",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
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


def tune_and_compare_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> Tuple[str, Any, Dict[str, float], Dict[str, Any]]:
    """
    Tune XGBoost with RandomizedSearchCV for both scale_pos_weight and SMOTE.
    
    Compares both tuned models using full CV scoring metrics and returns the best.
    """
    print("\n" + "=" * 75)
    print("XGBOOST HYPERPARAMETER TUNING & VARIANT COMPARISON (5-FOLD STRATIFIED CV)")
    print("=" * 75)

    scoring = ["roc_auc", "average_precision", "f1", "recall"]
    default_spw = float((y_train == 0).sum() / (y_train == 1).sum())

    # 1. Variant 1: scale_pos_weight
    print(f"[Tuning] Variant 1: scale_pos_weight (Baseline ratio: {default_spw:.2f})...")
    param_dist_spw = {
        "n_estimators": [100, 150, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.75, 0.85, 1.0],
        "colsample_bytree": [0.75, 0.85, 1.0],
        "gamma": [0, 0.1, 0.3],
        "scale_pos_weight": [default_spw * 0.8, default_spw, default_spw * 1.2],
    }
    xgb_spw_base = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    search_spw = RandomizedSearchCV(
        xgb_spw_base,
        param_distributions=param_dist_spw,
        n_iter=8,
        cv=cv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    search_spw.fit(X_train, y_train)
    best_xgb_spw = search_spw.best_estimator_
    best_params_spw = search_spw.best_params_

    cv_res_spw = cross_validate(best_xgb_spw, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    spw_metrics = {
        "cv_roc_auc_mean": float(np.mean(cv_res_spw["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cv_res_spw["test_roc_auc"])),
        "cv_pr_auc_mean": float(np.mean(cv_res_spw["test_average_precision"])),
        "cv_pr_auc_std": float(np.std(cv_res_spw["test_average_precision"])),
        "cv_f1_mean": float(np.mean(cv_res_spw["test_f1"])),
        "cv_recall_mean": float(np.mean(cv_res_spw["test_recall"])),
    }
    print(f"  Best params (scale_pos_weight): {best_params_spw}")
    print(
        f"  [scale_pos_weight] CV ROC-AUC: {spw_metrics['cv_roc_auc_mean']:.4f} (+/- {spw_metrics['cv_roc_auc_std']:.4f}) | "
        f"PR-AUC: {spw_metrics['cv_pr_auc_mean']:.4f} | F1: {spw_metrics['cv_f1_mean']:.4f}"
    )

    # 2. Variant 2: SMOTE Pipeline
    print("\n[Tuning] Variant 2: SMOTE Pipeline...")
    param_dist_smote = {
        "xgb__n_estimators": [100, 150, 200],
        "xgb__max_depth": [4, 6, 8],
        "xgb__learning_rate": [0.03, 0.05, 0.1],
        "xgb__subsample": [0.75, 0.85, 1.0],
        "xgb__colsample_bytree": [0.75, 0.85, 1.0],
        "xgb__gamma": [0, 0.1, 0.3],
    }
    pipe_smote_base = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("xgb", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)),
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

    # 3. Decision rule
    if spw_metrics["cv_roc_auc_mean"] >= smote_metrics["cv_roc_auc_mean"]:
        best_variant = "scale_pos_weight"
        best_model = best_xgb_spw
        best_cv_metrics = spw_metrics
        best_params = best_params_spw
    else:
        best_variant = "SMOTE"
        best_model = best_pipe_smote
        best_cv_metrics = smote_metrics
        best_params = best_params_smote

    print(f"\n--> Selected Superior XGBoost Variant: '{best_variant}'")
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


def append_xgb_to_comparison_csv(xgb_record: Dict[str, Any]) -> pd.DataFrame:
    """
    Append or update the XGBoost record in reports/model_comparison.csv
    without overwriting existing rows.
    """
    if COMPARISON_CSV_PATH.exists():
        df_comp = pd.read_csv(COMPARISON_CSV_PATH)
        df_comp = df_comp[df_comp["Model"] != "XGBoost"]
        df_updated = pd.concat([df_comp, pd.DataFrame([xgb_record])], ignore_index=True)
    else:
        df_updated = pd.DataFrame([xgb_record])

    df_updated.to_csv(COMPARISON_CSV_PATH, index=False)
    print(f"\n[Artifact] Updated comparison metrics saved to: {COMPARISON_CSV_PATH}")
    return df_updated


def plot_classical_roc_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """Generate publication-ready ROC curve comparison for all 4 classical models."""
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

    palette = {
        "Logistic Regression": "#7f7f7f",
        "Decision Tree": "#2ca02c",
        "Random Forest": "#1f77b4",
        "XGBoost": "#d62728",
    }
    line_styles = {
        "Logistic Regression": ":",
        "Decision Tree": "-.",
        "Random Forest": "--",
        "XGBoost": "-",
    }

    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc:.4f})",
            color=palette.get(name, "#333333"),
            linestyle=line_styles.get(name, "-"),
            linewidth=2.2 if name == "XGBoost" else 1.8,
        )

    # Reference random line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance (AUC = 0.5000)")

    ax.set_title("Classical ML Models: ROC Curve Comparison (Held-Out Test Set)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="medium")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="medium")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", frameon=True, fontsize=10.5)

    output_path = output_dir / "classical_roc_curves.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved ROC curve comparison to: {output_path}")


def plot_best_classical_confusion_matrix(
    model: Any,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """Generate detailed publication-ready confusion matrix for the top classical model."""
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        square=True,
        linewidths=1.5,
        linecolor="white",
        annot_kws={"size": 13, "weight": "bold"},
        xticklabels=["Non-Default (0)", "Default (1)"],
        yticklabels=["Non-Default (0)", "Default (1)"],
        ax=ax,
    )

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    ax.set_title(
        f"Confusion Matrix: {model_name} (Top Classical Model)\n"
        f"Accuracy: {acc:.2%} | Precision: {prec:.2%} | Recall: {rec:.2%} | F1: {f1:.4f}",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="medium", labelpad=10)
    ax.set_ylabel("True Ground Truth Label", fontsize=12, fontweight="medium", labelpad=10)

    output_path = output_dir / "best_classical_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved confusion matrix to: {output_path}")


def run_xgboost() -> pd.DataFrame:
    """Execute XGBoost training, evaluation, comparison, and summary visualizations."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("PHASE 2B-2: XGBOOST TRAINING & CLASSICAL ML WRAP-UP")
    print("=" * 75)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 2. Hyperparameter tuning & variant comparison
    best_variant, best_model, cv_metrics, best_params = tune_and_compare_xgb(
        X_train=X_train,
        y_train=y_train,
        cv=cv,
    )

    # 3. Refit on full training set
    print(f"\nRefitting optimal XGBoost ({best_variant}) on full training partition...")
    best_model.fit(X_train, y_train)

    # 4. Evaluate on test set
    test_metrics = evaluate_on_test_set(best_model, X_test, y_test)
    print("\n" + "=" * 75)
    print("TEST SET EVALUATION RESULTS (XGBOOST)")
    print("=" * 75)
    for metric_name, val in test_metrics.items():
        print(f"  {metric_name:<15}: {val:.4f}")

    # 5. Save model artifact
    xgb_model_path = MODELS_DIR / "xgboost.joblib"
    joblib.dump(best_model, xgb_model_path)
    print(f"\n[Artifact] Saved tuned XGBoost model to: {xgb_model_path}")

    # 6. Append to model_comparison.csv
    xgb_record = {
        "Model": "XGBoost",
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
    df_comparison = append_xgb_to_comparison_csv(xgb_record)

    # 7. Identify best overall classical model and save to models/best_classical_model.joblib
    # Primary criterion: Test ROC-AUC (and PR-AUC)
    best_classical_row = df_comparison.sort_values(
        by=["Test_ROC_AUC", "Test_PR_AUC"],
        ascending=False,
    ).iloc[0]
    best_model_name = best_classical_row["Model"]
    print(f"\n[Champion] Top Overall Classical Model: {best_model_name} (ROC-AUC: {best_classical_row['Test_ROC_AUC']:.4f})")

    model_file_map = {
        "Logistic Regression": MODELS_DIR / "logistic_regression.joblib",
        "Decision Tree": MODELS_DIR / "decision_tree.joblib",
        "Random Forest": MODELS_DIR / "random_forest.joblib",
        "XGBoost": MODELS_DIR / "xgboost.joblib",
    }
    best_classical_model = joblib.load(model_file_map[best_model_name])
    best_model_save_path = MODELS_DIR / "best_classical_model.joblib"
    joblib.dump(best_classical_model, best_model_save_path)
    print(f"[Artifact] Saved {best_model_name} as top classical model to: {best_model_save_path}")

    # 8. Load all 4 classical models to generate comparative figures
    all_models = {
        name: joblib.load(path) for name, path in model_file_map.items() if path.exists()
    }
    plot_classical_roc_curves(all_models, X_test, y_test, FIGURES_DIR)
    plot_best_classical_confusion_matrix(best_classical_model, best_model_name, X_test, y_test, FIGURES_DIR)

    # 9. Print final consolidated 4-model comparison table
    print("\n" + "=" * 110)
    print("CONSOLIDATED 4-MODEL CLASSICAL ML COMPARISON TABLE (HELD-OUT TEST SET)")
    print("=" * 110)
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
    print("=" * 110)

    return df_comparison


if __name__ == "__main__":
    run_xgboost()
