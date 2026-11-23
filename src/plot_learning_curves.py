from pathlib import Path
import sys
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, learning_curve

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_COMBINED_PATH = FIGURES_DIR / "learning_curves_classical_models.png"

# Model configurations: (display_label, model_filename, subplot_grid_coord, single_plot_filename)
MODEL_CONFIGS = [
    ("(a) Logistic Regression", "logistic_regression.joblib", (0, 0), "learning_curve_logistic_regression.png"),
    ("(b) Decision Tree", "decision_tree.joblib", (0, 1), "learning_curve_decision_tree.png"),
    ("(c) Random Forest", "random_forest.joblib", (1, 0), "learning_curve_random_forest.png"),
    ("(d) XGBoost", "xgboost.joblib", (1, 1), "learning_curve_xgboost.png"),
]


def load_training_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load processed training features and labels."""
    x_train_path = DATA_DIR / "X_train.csv"
    y_train_path = DATA_DIR / "y_train.csv"

    if not x_train_path.exists() or not y_train_path.exists():
        raise FileNotFoundError(f"Processed training data not found in {DATA_DIR}")

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel().astype(int)
    return X_train, y_train


def plot_learning_curves() -> None:
    """Compute and plot learning curves for all 4 classical models (both combined 2x2 and individual plots)."""
    print("Loading processed training datasets...")
    X_train, y_train = load_training_data()
    print(f"X_train shape: {X_train.shape} | y_train size: {len(y_train):,}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_size_fractions = np.linspace(0.1, 1.0, 5)

    fig_combined, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=False)
    fig_combined.suptitle(
        "Learning Curves for Classical ML Models (5-Fold Stratified CV)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for title, model_filename, (row, col), single_filename in MODEL_CONFIGS:
        model_path = MODELS_DIR / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        print(f"\nEvaluating learning curve for {title}...")
        model = joblib.load(model_path)

        train_sizes, train_scores, cv_scores = learning_curve(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring="accuracy",
            train_sizes=train_size_fractions,
            n_jobs=-1,
            random_state=42,
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        cv_mean = np.mean(cv_scores, axis=1)
        cv_std = np.std(cv_scores, axis=1)
        final_cv_acc = cv_mean[-1]

        # ── 1. Plot in Combined 2x2 Grid ─────────────────────────────────────
        ax = axes[row, col]

        # Red Training Accuracy line and shaded ±1 std band
        ax.plot(
            train_sizes,
            train_mean,
            color="#D9534F",
            marker="o",
            linewidth=2,
            label="Training Accuracy",
        )
        ax.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.18,
            color="#D9534F",
        )

        # Green Testing/CV Accuracy line and shaded ±1 std band
        ax.plot(
            train_sizes,
            cv_mean,
            color="#2E7D32",
            marker="s",
            linewidth=2,
            label="Testing / CV Accuracy",
        )
        ax.fill_between(
            train_sizes,
            cv_mean - cv_std,
            cv_mean + cv_std,
            alpha=0.18,
            color="#2E7D32",
        )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Training Examples", fontsize=11, fontweight="medium")
        ax.set_ylabel("Accuracy", fontsize=11, fontweight="medium")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=10)

        ax.annotate(
            f"Final CV: {final_cv_acc:.3%}",
            xy=(train_sizes[-1], cv_mean[-1]),
            xytext=(-70, -25),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2E7D32", alpha=0.9),
            fontsize=9,
            fontweight="semibold",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="#2E7D32"),
        )

        # ── 2. Plot Standalone Individual 300 DPI Figure ─────────────────────
        fig_single, ax_single = plt.subplots(figsize=(8.5, 6))

        ax_single.plot(
            train_sizes,
            train_mean,
            color="#D9534F",
            marker="o",
            linewidth=2,
            label="Training Accuracy",
        )
        ax_single.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.18,
            color="#D9534F",
        )

        ax_single.plot(
            train_sizes,
            cv_mean,
            color="#2E7D32",
            marker="s",
            linewidth=2,
            label="Testing / CV Accuracy",
        )
        ax_single.fill_between(
            train_sizes,
            cv_mean - cv_std,
            cv_mean + cv_std,
            alpha=0.18,
            color="#2E7D32",
        )

        clean_name = title.split(") ", 1)[-1]
        ax_single.set_title(f"Learning Curve — {clean_name} (5-Fold Stratified CV)", fontsize=13, fontweight="bold", pad=12)
        ax_single.set_xlabel("Training Examples", fontsize=11, fontweight="medium")
        ax_single.set_ylabel("Accuracy", fontsize=11, fontweight="medium")
        ax_single.grid(True, linestyle="--", alpha=0.5)
        ax_single.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=10)

        ax_single.annotate(
            f"Final CV: {final_cv_acc:.3%}",
            xy=(train_sizes[-1], cv_mean[-1]),
            xytext=(-70, -25),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2E7D32", alpha=0.9),
            fontsize=9,
            fontweight="semibold",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="#2E7D32"),
        )

        single_out_path = FIGURES_DIR / single_filename
        fig_single.tight_layout()
        fig_single.savefig(single_out_path, dpi=300, bbox_inches="tight")
        plt.close(fig_single)
        print(f"  -> Saved individual plot: {single_out_path} (300 DPI)")

    # Save combined 2x2 grid
    fig_combined.tight_layout(rect=[0, 0, 1, 0.96])
    fig_combined.savefig(OUTPUT_COMBINED_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_combined)
    print(f"\nCombined learning curves plot successfully saved to: {OUTPUT_COMBINED_PATH} (300 DPI)")


if __name__ == "__main__":
    plot_learning_curves()
