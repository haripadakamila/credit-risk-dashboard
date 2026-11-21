import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join("data", "processed")
MODEL_DIR  = "models"
FIGURES_DIR = os.path.join("reports", "figures")
REPORT_CSV  = os.path.join("reports", "model_comparison.csv")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CHECKPOINT = os.path.join(MODEL_DIR, "mlp_pytorch.pt")

# ── Load processed data ──────────────────────────────────────────────────────
X_train_full = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv")).values.astype(np.float32)
X_test       = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv")).values.astype(np.float32)
y_train_full = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel().astype(np.float32)
y_test       = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel().astype(np.float32)

print(f"Loaded  X_train_full: {X_train_full.shape},  X_test: {X_test.shape}")

# ── Stratified 15% validation split (from training set only) ─────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.15,
    random_state=SEED,
    stratify=y_train_full
)
print(f"Train split : {X_tr.shape}   Val split: {X_val.shape}")

# ── pos_weight computed from training split only ──────────────────────────────
n_neg = (y_tr == 0).sum()
n_pos = (y_tr == 1).sum()
pos_weight_val = float(n_neg) / float(n_pos)
print(f"pos_weight  : {pos_weight_val:.4f}  (neg={int(n_neg)}, pos={int(n_pos)})")

# ── DataLoaders ───────────────────────────────────────────────────────────────
BATCH_SIZE = 256

def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_tr,  y_tr,  shuffle=True)
val_loader   = make_loader(X_val, y_val, shuffle=False)
test_loader  = make_loader(X_test, y_test, shuffle=False)

# ── Model definition ──────────────────────────────────────────────────────────
class CreditRiskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))   # raw logit — no sigmoid here
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


INPUT_DIM = X_tr.shape[1]
model = CreditRiskMLP(input_dim=INPUT_DIM)
print(f"\nModel architecture:\n{model}\n")

# ── Loss / Optimizer / Scheduler ─────────────────────────────────────────────
pos_weight_tensor = torch.tensor([pos_weight_val])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)

# ── Training loop with early stopping ────────────────────────────────────────
MAX_EPOCHS    = 100
ES_PATIENCE   = 15

best_val_auc  = -np.inf
epochs_no_imp = 0

history = {"epoch": [], "train_loss": [], "val_loss": [], "val_auc": []}

for epoch in range(1, MAX_EPOCHS + 1):
    # -- train --
    model.train()
    train_loss = 0.0
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(yb)
    train_loss /= len(y_tr)

    # -- validate --
    model.eval()
    val_probs  = []
    val_labels = []
    val_loss   = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            logits = model(Xb)
            loss_v = criterion(logits, yb)
            val_loss += loss_v.item() * len(yb)
            probs  = torch.sigmoid(logits).numpy()
            val_probs.extend(probs)
            val_labels.extend(yb.numpy())
    val_loss /= len(y_val)

    val_auc = roc_auc_score(val_labels, val_probs)
    scheduler.step(val_auc)

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_auc"].append(val_auc)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_ROC-AUC={val_auc:.4f}")

    # -- checkpoint if improved --
    if val_auc > best_val_auc:
        best_val_auc  = val_auc
        epochs_no_imp = 0
        torch.save(model.state_dict(), CHECKPOINT)
    else:
        epochs_no_imp += 1
        if epochs_no_imp >= ES_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {ES_PATIENCE} epochs)")
            break

print(f"\nBest validation ROC-AUC : {best_val_auc:.4f}")
print(f"Model checkpoint saved  : {CHECKPOINT}")

# ── Load best weights and evaluate on test set ───────────────────────────────
model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
model.eval()

test_probs  = []
test_labels = []
with torch.no_grad():
    for Xb, yb in test_loader:
        logits = model(Xb)
        probs  = torch.sigmoid(logits).numpy()
        test_probs.extend(probs)
        test_labels.extend(yb.numpy())

test_probs  = np.array(test_probs)
test_labels = np.array(test_labels)
test_preds  = (test_probs >= 0.5).astype(int)

acc      = accuracy_score(test_labels, test_preds)
prec     = precision_score(test_labels, test_preds, zero_division=0)
rec      = recall_score(test_labels, test_preds, zero_division=0)
f1       = f1_score(test_labels, test_preds, zero_division=0)
roc_auc  = roc_auc_score(test_labels, test_probs)
pr_auc   = average_precision_score(test_labels, test_probs)
brier    = brier_score_loss(test_labels, test_probs)

print("\n" + "=" * 55)
print("  PyTorch MLP — Test Set Metrics")
print("=" * 55)
print(f"  Accuracy   : {acc * 100:.2f}%")
print(f"  Precision  : {prec * 100:.2f}%")
print(f"  Recall     : {rec * 100:.2f}%")
print(f"  F1-score   : {f1:.4f}")
print(f"  ROC-AUC    : {roc_auc:.4f}")
print(f"  PR-AUC     : {pr_auc:.4f}")
print(f"  Brier Score: {brier:.4f}")
print("=" * 55)

# ── Training curves ───────────────────────────────────────────────────────────
epochs_arr = history["epoch"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("PyTorch MLP — Training Curves", fontsize=14, fontweight="bold")

# Loss curves
ax1.plot(epochs_arr, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
ax1.plot(epochs_arr, history["val_loss"],   label="Val Loss",   color="#ff7f0e", linewidth=2, linestyle="--")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("BCEWithLogitsLoss")
ax1.set_title("Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# ROC-AUC curve
ax2.plot(epochs_arr, history["val_auc"], label="Val ROC-AUC", color="#2ca02c", linewidth=2)
best_epoch_idx = int(np.argmax(history["val_auc"]))
ax2.axvline(x=epochs_arr[best_epoch_idx], color="red", linestyle=":", linewidth=1.5,
            label=f"Best epoch {epochs_arr[best_epoch_idx]} (AUC={history['val_auc'][best_epoch_idx]:.4f})")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("ROC-AUC")
ax2.set_title("Validation ROC-AUC")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
curves_path = os.path.join(FIGURES_DIR, "mlp_training_curves.png")
plt.savefig(curves_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nTraining curves saved : {curves_path}")

# ── Append metrics to model_comparison.csv ───────────────────────────────────
new_row = pd.DataFrame([{
    "Model":            "PyTorch MLP",
    "Best_Variant":     "BCEWithLogitsLoss+pos_weight",
    "CV_ROC_AUC":       float("nan"),   # no CV was run for the MLP
    "CV_PR_AUC":        float("nan"),
    "CV_F1":            float("nan"),
    "Test_Accuracy":    acc,
    "Test_Precision":   prec,
    "Test_Recall":      rec,
    "Test_F1":          f1,
    "Test_ROC_AUC":     roc_auc,
    "Test_PR_AUC":      pr_auc,
    "Test_Brier_Score": brier,
}])

existing = pd.read_csv(REPORT_CSV)
# Guard: remove any stale MLP row before appending
existing = existing[existing["Model"] != "PyTorch MLP"]
updated  = pd.concat([existing, new_row], ignore_index=True)
updated.to_csv(REPORT_CSV, index=False)

print(f"model_comparison.csv updated : {REPORT_CSV}")
print(f"\nFull comparison table ({len(updated)} models):")
print(updated[["Model", "Test_Accuracy", "Test_Precision", "Test_Recall",
               "Test_F1", "Test_ROC_AUC", "Test_PR_AUC", "Test_Brier_Score"]].to_string(index=False))

