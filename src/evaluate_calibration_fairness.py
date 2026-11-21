import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)

# ── Paths & Environment ──────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_PATH = ROOT_DIR / "models" / "xgboost.joblib"
PREPROCESSOR_PATH = ROOT_DIR / "models" / "preprocessor.joblib"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
REPORTS_DIR = ROOT_DIR / "reports"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_PLOT_PATH = FIGURES_DIR / "calibration_curve.png"
FAIRNESS_CSV_PATH = REPORTS_DIR / "fairness_audit.csv"

# ── Load Model, Preprocessor, and Test Data ──────────────────────────────────
print(f"Loading champion XGBoost model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

print(f"Loading preprocessor from: {PREPROCESSOR_PATH}")
preprocessor = joblib.load(PREPROCESSOR_PATH)

print(f"Loading untouched test dataset from: {DATA_DIR}")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel().astype(int)

print(f"Test samples: {len(y_test):,} | Defaults: {int(y_test.sum()):,} ({y_test.mean():.2%})")

# ── Model Predictions ────────────────────────────────────────────────────────
y_probs = model.predict_proba(X_test)[:, 1]
y_preds = (y_probs >= 0.5).astype(int)
brier = brier_score_loss(y_test, y_probs)

print(f"Overall Test Brier Score: {brier:.4f}")

# ── 1. Calibration Curve ─────────────────────────────────────────────────────
print("\nComputing calibration curve (10 uniform bins)...")
prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10, strategy="uniform")

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8.5, 7.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
)

# Main reliability curve
ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
ax1.plot(
    prob_pred,
    prob_true,
    marker="o",
    linewidth=2,
    color="#1f77b4",
    label=f"XGBoost (Brier Score = {brier:.4f})",
)
ax1.set_ylabel("Empirical Default Rate", fontsize=11)
ax1.set_title("Reliability / Calibration Curve — Champion XGBoost", fontsize=13, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper left", fontsize=10)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)

# Probability distribution histogram
ax2.hist(y_probs, bins=25, color="#1f77b4", edgecolor="black", alpha=0.7)
ax2.set_xlabel("Mean Predicted Default Probability", fontsize=11)
ax2.set_ylabel("Count", fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.set_xlim(-0.02, 1.02)

plt.tight_layout()
plt.savefig(CALIBRATION_PLOT_PATH, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Calibration curve saved: {CALIBRATION_PLOT_PATH}")

# ── 2. Group Fairness Audit ──────────────────────────────────────────────────
print("\nExecuting Group Fairness Audit across Age Groups and Home Ownership...")

# Reconstruct unscaled age using fitted scaler statistics
age_mean = preprocessor.scaler_.mean_[0]
age_scale = preprocessor.scaler_.scale_[0]
age_unscaled = np.round(X_test["person_age"] * age_scale + age_mean)

age_bins = [0, 25, 35, 50, 150]
age_labels = ["20-25", "26-35", "36-50", "50+"]
age_groups = pd.cut(age_unscaled, bins=age_bins, labels=age_labels)

# Reconstruct home ownership from one-hot columns
home_ownership_cols = [
    "person_home_ownership_RENT",
    "person_home_ownership_MORTGAGE",
    "person_home_ownership_OWN",
    "person_home_ownership_OTHER",
]
home_ownership_series = pd.Series(index=X_test.index, dtype=str)
for col in home_ownership_cols:
    val_name = col.replace("person_home_ownership_", "")
    home_ownership_series[X_test[col] == 1] = val_name

# Helper to compute subgroup metrics
def evaluate_subgroup(y_true_sub, y_pred_sub, y_prob_sub):
    acc = accuracy_score(y_true_sub, y_pred_sub)
    prec = precision_score(y_true_sub, y_pred_sub, zero_division=0)
    rec = recall_score(y_true_sub, y_pred_sub, zero_division=0)
    f1 = f1_score(y_true_sub, y_pred_sub, zero_division=0)
    # ROC-AUC requires at least one positive and one negative sample
    if len(np.unique(y_true_sub)) > 1:
        auc = roc_auc_score(y_true_sub, y_prob_sub)
    else:
        auc = float("nan")
    return acc, prec, rec, f1, auc

records = []

# Audit: Age Groups
for grp in age_labels:
    mask = (age_groups == grp).values
    y_sub = y_test[mask]
    pred_sub = y_preds[mask]
    prob_sub = y_probs[mask]
    acc, prec, rec, f1, auc = evaluate_subgroup(y_sub, pred_sub, prob_sub)
    records.append({
        "Attribute": "Age Group",
        "Subgroup": grp,
        "Sample_Count": int(len(y_sub)),
        "Default_Count": int(y_sub.sum()),
        "Default_Rate": float(y_sub.mean()),
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1_Score": float(f1),
        "ROC_AUC": float(auc),
    })

# Audit: Home Ownership
for grp in ["RENT", "MORTGAGE", "OWN", "OTHER"]:
    mask = (home_ownership_series == grp).values
    y_sub = y_test[mask]
    pred_sub = y_preds[mask]
    prob_sub = y_probs[mask]
    acc, prec, rec, f1, auc = evaluate_subgroup(y_sub, pred_sub, prob_sub)
    records.append({
        "Attribute": "Home Ownership",
        "Subgroup": grp,
        "Sample_Count": int(len(y_sub)),
        "Default_Count": int(y_sub.sum()),
        "Default_Rate": float(y_sub.mean()),
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1_Score": float(f1),
        "ROC_AUC": float(auc),
    })

# Overall benchmark row
acc_all, prec_all, rec_all, f1_all, auc_all = evaluate_subgroup(y_test, y_preds, y_probs)
records.append({
    "Attribute": "Overall",
    "Subgroup": "All Test Applicants",
    "Sample_Count": int(len(y_test)),
    "Default_Count": int(y_test.sum()),
    "Default_Rate": float(y_test.mean()),
    "Accuracy": float(acc_all),
    "Precision": float(prec_all),
    "Recall": float(rec_all),
    "F1_Score": float(f1_all),
    "ROC_AUC": float(auc_all),
})

fairness_df = pd.DataFrame(records)
fairness_df.to_csv(FAIRNESS_CSV_PATH, index=False)
print(f"Fairness audit report saved: {FAIRNESS_CSV_PATH}")

# ── Print Formatted Summary Table ────────────────────────────────────────────
print("\n" + "=" * 95)
print("  GROUP FAIRNESS & SUBGROUP PERFORMANCE AUDIT — CHAMPION XGBOOST")
print("=" * 95)

display_df = fairness_df.copy()
display_df["Default_Rate"] = display_df["Default_Rate"].apply(lambda x: f"{x * 100:.1f}%")
display_df["Accuracy"] = display_df["Accuracy"].apply(lambda x: f"{x * 100:.2f}%")
display_df["Precision"] = display_df["Precision"].apply(lambda x: f"{x * 100:.2f}%")
display_df["Recall"] = display_df["Recall"].apply(lambda x: f"{x * 100:.2f}%")
display_df["F1_Score"] = display_df["F1_Score"].apply(lambda x: f"{x:.4f}")
display_df["ROC_AUC"] = display_df["ROC_AUC"].apply(lambda x: f"{x:.4f}")

print(display_df.to_string(index=False))
print("=" * 95)
