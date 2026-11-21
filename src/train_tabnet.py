import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from pytorch_tabnet.tab_model import TabNetClassifier

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join("data", "processed")
MODEL_DIR = "models"
FIGURES_DIR = os.path.join("reports", "figures")
REPORT_CSV = os.path.join("reports", "model_comparison.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "tabnet_model")

# ── Load processed data ──────────────────────────────────────────────────────
X_train_df = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
feature_names = list(X_train_df.columns)
X_train_full = X_train_df.values.astype(np.float32)

X_test_df = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
X_test = X_test_df.values.astype(np.float32)

y_train_full = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel().astype(np.int64)
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel().astype(np.int64)

print(f"Loaded X_train_full: {X_train_full.shape}, X_test: {X_test.shape}")

# ── Stratified 15% validation split (training set only) ──────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.15,
    random_state=SEED,
    stratify=y_train_full,
)
print(f"Train split : {X_tr.shape} | Val split: {X_val.shape}")
print(f"Train target counts: 0={int((y_tr == 0).sum())}, 1={int((y_tr == 1).sum())}")

# ── Initialize TabNetClassifier ──────────────────────────────────────────────
clf = TabNetClassifier(
    seed=SEED,
    verbose=10,
)

# ── Train model with early stopping on validation ROC-AUC ────────────────────
print("\nTraining TabNetClassifier with weights=1 for balanced batch sampling...")
clf.fit(
    X_train=X_tr,
    y_train=y_tr,
    eval_set=[(X_val, y_val)],
    eval_name=["val"],
    eval_metric=["auc"],
    max_epochs=100,
    patience=15,
    batch_size=1024,
    virtual_batch_size=128,
    weights=1,
    drop_last=False,
)

# ── Save model ───────────────────────────────────────────────────────────────
saved_filepath = clf.save_model(MODEL_SAVE_PATH)
print(f"\nModel saved successfully: {saved_filepath}")

# ── Evaluate on untouched test set ───────────────────────────────────────────
test_probs = clf.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

acc = accuracy_score(y_test, test_preds)
prec = precision_score(y_test, test_preds, zero_division=0)
rec = recall_score(y_test, test_preds, zero_division=0)
f1 = f1_score(y_test, test_preds, zero_division=0)
roc_auc = roc_auc_score(y_test, test_probs)
pr_auc = average_precision_score(y_test, test_probs)
brier = brier_score_loss(y_test, test_probs)

print("\n" + "=" * 55)
print("  TabNet Classifier — Test Set Metrics")
print("=" * 55)
print(f"  Accuracy   : {acc * 100:.2f}%")
print(f"  Precision  : {prec * 100:.2f}%")
print(f"  Recall     : {rec * 100:.2f}%")
print(f"  F1-score   : {f1:.4f}")
print(f"  ROC-AUC    : {roc_auc:.4f}")
print(f"  PR-AUC     : {pr_auc:.4f}")
print(f"  Brier Score: {brier:.4f}")
print("=" * 55)

# ── Plot and save top 15 TabNet feature importances ──────────────────────────
importances = clf.feature_importances_
feat_imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances,
}).sort_values(by="importance", ascending=True)

top15_df = feat_imp_df.tail(15)

plt.figure(figsize=(10, 6))
plt.barh(top15_df["feature"], top15_df["importance"], color="#1f77b4")
plt.xlabel("Global Feature Importance", fontsize=11)
plt.title("Top 15 TabNet Feature Importances", fontsize=13, fontweight="bold")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()

feat_imp_path = os.path.join(FIGURES_DIR, "tabnet_feature_importance.png")
plt.savefig(feat_imp_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFeature importance plot saved: {feat_imp_path}")

# ── Append metrics to model_comparison.csv ───────────────────────────────────
new_row = pd.DataFrame([{
    "Model": "TabNet",
    "Best_Variant": "weights=1 (balanced)",
    "CV_ROC_AUC": float("nan"),
    "CV_PR_AUC": float("nan"),
    "CV_F1": float("nan"),
    "Test_Accuracy": acc,
    "Test_Precision": prec,
    "Test_Recall": rec,
    "Test_F1": f1,
    "Test_ROC_AUC": roc_auc,
    "Test_PR_AUC": pr_auc,
    "Test_Brier_Score": brier,
}])

existing = pd.read_csv(REPORT_CSV)
existing = existing[~existing["Model"].isin(["TabNet", "TabNet Classifier"])]
updated = pd.concat([existing, new_row], ignore_index=True)
updated.to_csv(REPORT_CSV, index=False)

print(f"\nmodel_comparison.csv updated: {REPORT_CSV}")
print(f"\nFull comparison table ({len(updated)} models):")
print(
    updated[
        [
            "Model",
            "Best_Variant",
            "Test_Accuracy",
            "Test_Precision",
            "Test_Recall",
            "Test_F1",
            "Test_ROC_AUC",
            "Test_PR_AUC",
            "Test_Brier_Score",
        ]
    ].to_string(index=False)
)
