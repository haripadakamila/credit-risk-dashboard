import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join("data", "processed")
MODEL_PATH = os.path.join("models", "xgboost.joblib")
FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

BEESWARM_PATH = os.path.join(FIGURES_DIR, "shap_summary_beeswarm.png")
BAR_PATH = os.path.join(FIGURES_DIR, "shap_summary_bar.png")
WATERFALL_HIGH_RISK_PATH = os.path.join(FIGURES_DIR, "shap_waterfall_high_risk.png")
WATERFALL_LOW_RISK_PATH = os.path.join(FIGURES_DIR, "shap_waterfall_low_risk.png")

# ── Load Model and Test Data ─────────────────────────────────────────────────
print(f"Loading champion XGBoost model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

print(f"Loading test features and targets from: {DATA_DIR}")
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()
print(f"Test feature matrix shape: {X_test.shape}")

# ── Compute SHAP Values ──────────────────────────────────────────────────────
print("\nInitializing shap.TreeExplainer on XGBoost estimator...")
explainer = shap.TreeExplainer(model)

print("Computing SHAP values across all test samples...")
shap_values = explainer(X_test)

# ── 1. SHAP Beeswarm Summary Plot (Top 15 Features) ──────────────────────────
print("\nGenerating SHAP beeswarm summary plot...")
plt.figure(figsize=(11, 7.5))
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title("SHAP Beeswarm Summary — XGBoost (Top 15 Features)", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(BEESWARM_PATH, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {BEESWARM_PATH}")

# ── 2. SHAP Mean |SHAP Value| Bar Plot (Top 15 Features) ─────────────────────
print("Generating SHAP feature importance bar plot...")
plt.figure(figsize=(10, 6.5))
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title("Mean |SHAP Value| Feature Importance — XGBoost (Top 15 Features)", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(BAR_PATH, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {BAR_PATH}")

# ── 3. Local SHAP Waterfall Plots ────────────────────────────────────────────
print("\nIdentifying representative high-risk and low-risk test samples...")
test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

# Candidate 1: High risk (correctly predicted Default with maximum probability)
high_risk_candidates = np.where((test_preds == 1) & (y_test == 1))[0]
high_risk_idx = high_risk_candidates[np.argmax(test_probs[high_risk_candidates])]

# Candidate 2: Low risk (correctly predicted Non-Default with minimum probability)
low_risk_candidates = np.where((test_preds == 0) & (y_test == 0))[0]
low_risk_idx = low_risk_candidates[np.argmin(test_probs[low_risk_candidates])]

print(f"High risk sample: index {high_risk_idx} | True Label: {y_test[high_risk_idx]} | Predicted Prob: {test_probs[high_risk_idx]:.4f}")
print(f"Low risk sample : index {low_risk_idx}  | True Label: {y_test[low_risk_idx]} | Predicted Prob: {test_probs[low_risk_idx]:.4f}")

# Plot High Risk Waterfall
plt.figure(figsize=(10, 6.5))
shap.plots.waterfall(shap_values[high_risk_idx], max_display=12, show=False)
plt.title(
    f"SHAP Waterfall — High Risk Default (Sample #{high_risk_idx}, Risk: {test_probs[high_risk_idx]*100:.1f}%)",
    fontsize=12,
    fontweight="bold",
    pad=15,
)
plt.tight_layout()
plt.savefig(WATERFALL_HIGH_RISK_PATH, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {WATERFALL_HIGH_RISK_PATH}")

# Plot Low Risk Waterfall
plt.figure(figsize=(10, 6.5))
shap.plots.waterfall(shap_values[low_risk_idx], max_display=12, show=False)
plt.title(
    f"SHAP Waterfall — Low Risk Non-Default (Sample #{low_risk_idx}, Risk: {test_probs[low_risk_idx]*100:.2f}%)",
    fontsize=12,
    fontweight="bold",
    pad=15,
)
plt.tight_layout()
plt.savefig(WATERFALL_LOW_RISK_PATH, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {WATERFALL_LOW_RISK_PATH}")

# ── Print Top Features by Mean Absolute SHAP ─────────────────────────────────
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean_|SHAP|": mean_abs_shap,
}).sort_values(by="Mean_|SHAP|", ascending=False).reset_index(drop=True)

print("\n" + "=" * 55)
print("  Top 10 Most Impactful Features (Mean |SHAP Value|)")
print("=" * 55)
for idx, row in shap_importance_df.head(10).iterrows():
    print(f"  {idx + 1:2d}. {row['Feature']:<30} : {row['Mean_|SHAP|']:.4f}")
print("=" * 55)
