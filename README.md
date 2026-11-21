# 💳 End-to-End Credit Risk Prediction & Underwriting Intelligence

An enterprise-grade, explainable machine learning system designed to predict credit default risk from loan applicant data. Built with strict data-leakage controls, calibrated probability estimates, deep neural architectures, comprehensive SHAP interpretability, and subgroup fairness auditing.

The system serves as a **human-in-the-loop decision-support engine** for credit risk officers and underwriters, facilitating tiered risk categorization (Low, Medium, High Risk) with actionable mitigation recommendations rather than automated rejection.

---

## 📌 Project Overview & Problem Statement

Credit scoring and loan default estimation are critical pillars of consumer lending. Predicting whether a borrower will default on a credit facility is challenging due to:
- **Severe Class Imbalance:** Default instances represent ~21.9% of total historical loans.
- **Strict Data Leakage Constraints:** Imputations, scalers, and encodings must be strictly fitted on training splits only.
- **Explainability & Compliance Mandates:** Financial regulations require model decisions to be transparent, auditable, and interpretable at both portfolio and applicant levels.
- **Ethical Lending & Human Oversight:** The model acts strictly as a decision aid. Marginal or elevated risk profiles trigger structured underwriter reviews, restructuring options, or guarantor requests—preventing unilateral automated denial.

---

## 📂 Project Structure

```text
credit-risk-prediction/
├── app/
│   └── streamlit_app.py                # Multi-tab interactive Streamlit underwriting dashboard
├── data/
│   ├── raw/
│   │   └── credit_risk_dataset.csv      # Raw historical loan applicant records (32,581 rows)
│   └── processed/
│       ├── train.csv, test.csv          # Stratified train/test splits with target
│       ├── X_train.csv, X_test.csv      # 29 scaled and one-hot encoded features
│       └── y_train.csv, y_test.csv      # Target partitions (loan_status: 0 vs 1)
├── models/
│   ├── preprocessor.joblib              # Fitted leakage-free CreditRiskPreprocessor
│   ├── logistic_regression.joblib       # Baseline Logistic Regression
│   ├── decision_tree.joblib             # Baseline Decision Tree
│   ├── random_forest.joblib             # Tuned Random Forest Classifier
│   ├── xgboost.joblib                   # Champion XGBoost Classifier
│   ├── best_classical_model.joblib      # Copy of champion classical model
│   ├── mlp_pytorch.pt                   # Checkpoint of tabular PyTorch MLP
│   └── tabnet_model.zip                 # Trained PyTorch-TabNet attention model
├── reports/
│   ├── model_comparison.csv             # Standardized benchmark metrics across all 6 models
│   ├── fairness_audit.csv               # Demographic subgroup performance and fairness audit
│   └── figures/                         # Publication-ready diagnostic charts
│       ├── target_distribution.png
│       ├── default_rate_by_grade.png
│       ├── default_rate_by_intent.png
│       ├── default_rate_by_home_ownership.png
│       ├── correlation_heatmap.png
│       ├── classical_roc_curves.png
│       ├── best_classical_confusion_matrix.png
│       ├── mlp_training_curves.png
│       ├── tabnet_feature_importance.png
│       ├── shap_summary_beeswarm.png
│       ├── shap_summary_bar.png
│       ├── shap_waterfall_high_risk.png
│       ├── shap_waterfall_low_risk.png
│       └── calibration_curve.png
├── src/
│   ├── __init__.py
│   ├── eda.py                          # Exploratory data analysis & statistical plotting
│   ├── preprocess.py                   # Leakage-free preprocessing, cleaning, & ratio engineering
│   ├── train_baselines.py              # Logistic Regression & Decision Tree (CV + SMOTE/weighting)
│   ├── train_rf.py                     # Random Forest tuning & cross-validation
│   ├── train_xgb.py                    # XGBoost tuning & champion model export
│   ├── train_mlp.py                    # Tabular PyTorch MLP with early stopping & learning curves
│   ├── train_tabnet.py                 # PyTorch TabNet training & attention feature importance
│   ├── explain_shap.py                 # Global and local SHAP explanation generation
│   └── evaluate_calibration_fairness.py # Calibration curve & demographic group fairness audit
├── tests/                              # Unit & pipeline test directory
├── requirements.txt                    # Project dependency manifest
└── README.md                           # Documentation
```

---

## 🔬 Methodology Summary

### 1. Leakage-Free Preprocessing & Integrity Cleaning
- **Integrity Cleansing:** Dropped exact duplicates and filtered biological/physical impossibilities (`person_age > 100`, `person_emp_length > 60`).
- **Zero-Leakage Stratified Split:** Partitioned into an 80/20 train-test split (`random_state=42`) *before* any feature transformation.
- **Median Imputation:** Missing employment lengths (`person_emp_length`) and interest rates (`loan_int_rate`) were imputed strictly using training set medians (`4.0` years and `10.99%`).
- **Feature Standardization:** Numerical features were normalized using `StandardScaler` fitted only on training data; categorical variables were encoded using `OneHotEncoder(handle_unknown="ignore")`.

### 2. Domain Banking Ratio Feature Engineering
To capture applicant solvency and debt liability beyond raw dollar amounts, 3 core banking ratios were engineered:
- **Loan-to-Income Ratio:** $\text{loan\_amnt} / \text{person\_income}$
- **Interest Burden to Income:** $(\text{loan\_amnt} \times (\text{loan\_int\_rate} / 100)) / \text{person\_income}$
- **Credit History to Age Ratio:** $\text{cb\_person\_cred\_hist\_length} / \text{person\_age}$

### 3. Class Imbalance Mitigation
The dataset exhibits significant imbalance (~78.1% non-default vs. ~21.9% default). Strategies evaluated:
- **Inside Pipeline SMOTE:** Evaluated strictly inside training folds to prevent validation leakage.
- **Loss Reweighting:** `class_weight="balanced"` for linear and tree baselines, `scale_pos_weight` ($N_{\text{neg}} / N_{\text{pos}} \approx 2.86$) for XGBoost, `pos_weight = 3.57` inside PyTorch `BCEWithLogitsLoss`, and `weights=1` balanced batch sampling for TabNet.

### 4. 6-Model Benchmark Suite
Six model families were trained and systematically benchmarked on identical, untouched test partitions:
1. **Logistic Regression** (L2-penalized baseline)
2. **Decision Tree** (Cost-complexity pruned baseline)
3. **Random Forest** (150 estimators, tuned max depth and split thresholds)
4. **XGBoost** (Gradient boosted trees with subsampling, gamma, and depth regularization)
5. **PyTorch MLP** (4-layer tabular deep neural network: Linear $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout)
6. **PyTorch TabNet** (Attentive transformer with sparsemax sequential decision steps)

### 5. Explainability & Model Interpretability (SHAP)
- **Global Explanations:** TreeExplainer computed across the test partition reveals that `person_income`, `loan_to_income_ratio`, `loan_int_rate`, and `person_home_ownership_RENT` are the primary drivers of portfolio default risk.
- **Local Explanations:** Real-time SHAP waterfall decompositions break down individual applicant predictions into positive (+ risk) and negative (- risk) log-odds contributions.

### 6. Calibration & Fairness Auditing
- **Probability Calibration:** Reliability curve analysis with a Brier Score of **0.0556** demonstrates strong empirical probability alignment across all 10 risk deciles.
- **Demographic Group Fairness:** Evaluated model parity across **Age Groups** (20–25, 26–35, 36–50, 50+) and **Home Ownership** (Rent, Mortgage, Own, Other), demonstrating consistent accuracy (>92%) and ROC-AUC (>0.89) across all cohorts.

---

## 📊 Model Performance Benchmark

Evaluated strictly on the untouched test partition ($N = 6,482$ records):

| Model Architecture | Class Balance Strategy | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Test PR-AUC | Test Brier Score |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | `class_weight="balanced"` | 80.96% | 54.45% | **79.41%** | 0.6460 | 0.8819 | 0.7359 | 0.1353 |
| **Decision Tree** | `class_weight="balanced"` | 90.56% | 80.44% | 75.11% | 0.7768 | 0.9097 | 0.8542 | 0.0849 |
| **Random Forest** | `class_weight="balanced"` | 92.16% | 85.66% | 77.08% | 0.8114 | 0.9395 | 0.8947 | 0.0671 |
| **XGBoost (Champion)** 🏆 | `scale_pos_weight` | **93.35%** | **89.70%** | 78.63% | **0.8380** | **0.9491** | **0.9076** | **0.0556** |
| **PyTorch MLP** | `BCEWithLogitsLoss+pos_weight` | 89.46% | 75.75% | 76.23% | 0.7599 | 0.9253 | 0.8682 | 0.0906 |
| **TabNet Classifier** | `weights=1 (balanced)` | 90.40% | 79.66% | 75.39% | 0.7746 | 0.9234 | 0.8670 | 0.0863 |

> **Champion Selection:** **XGBoost** achieved the highest discrimination (ROC-AUC: **0.9491**, PR-AUC: **0.9076**), highest precision (**89.70%**), and lowest Brier score (**0.0556**), minimizing both false positives and missed defaults.

---

## 🛠️ Setup & How to Run

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.14 on Windows)
- Git

### 2. Environment Setup
Clone the repository and initialize the virtual environment:

```bash
# Clone the repository
git clone <repo-url>
cd credit-risk-prediction

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install CPU PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install all project dependencies
pip install -r requirements.txt
```

### 3. Running the Pipeline Step-by-Step

Execute the pipeline modules sequentially using the virtual environment interpreter:

```bash
# 1. Exploratory Data Analysis & statistical visualization
python src/eda.py

# 2. Leakage-free preprocessing, feature engineering, & dataset export
python src/preprocess.py

# 3. Train classical baselines (Logistic Regression & Decision Tree)
python src/train_baselines.py

# 4. Train & tune Random Forest
python src/train_rf.py

# 5. Train & tune champion XGBoost and export summary plots
python src/train_xgb.py

# 6. Train PyTorch Multi-Layer Perceptron (MLP)
python src/train_mlp.py

# 7. Train PyTorch TabNet with attention mechanisms
python src/train_tabnet.py

# 8. Generate global and local SHAP explanations
python src/explain_shap.py

# 9. Evaluate probability calibration and demographic group fairness
python src/evaluate_calibration_fairness.py
```

### 4. Launching the Interactive Streamlit Dashboard

Run the Streamlit application to access the multi-tab underwriting interface:

```bash
streamlit run app/streamlit_app.py
```

Open your browser at **`http://localhost:8501`** to explore:
- **Tab 1: 🎯 Live Credit Risk Predictor:** Interactive applicant input form, instant default probability, risk tier badge, underwriting guidance, and live SHAP waterfall plot.
- **Tab 2: 📈 Model Comparison:** Full 6-model benchmark table, classical ROC curves, confusion matrix, MLP learning curves, and TabNet feature importances.
- **Tab 3: 🌐 Global Insights & SHAP:** Global SHAP beeswarm plots, mean |SHAP| bar charts, default rate trends by credit grade, and feature correlation heatmaps.

---

## ⚖️ Ethical AI & Governance Note

This credit risk engine is designed strictly as a **decision-support tool** to empower credit officers with transparent, calibrated predictions and risk-driver attributions. It should **never** be used for automated unilateral rejections. Underwriters must cross-reference model outputs with institutional lending policies, applicant disclosures, and qualitative context.
