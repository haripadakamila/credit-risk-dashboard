import io
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import streamlit as st

# ── Ensure project root is in sys.path ────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ── Configuration & Page Setup ───────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Intelligence Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Friendly display names for SHAP features
FEATURE_NAME_MAP = {
    "person_age": "Applicant Age",
    "person_income": "Annual Income",
    "person_emp_length": "Employment Length",
    "loan_amnt": "Loan Amount",
    "loan_int_rate": "Interest Rate",
    "loan_percent_income": "Loan as % of Income",
    "cb_person_cred_hist_length": "Credit History Length",
    "loan_to_income_ratio": "Loan-to-Income Ratio",
    "interest_to_income_burden": "Interest Burden / Income",
    "credit_history_to_age_ratio": "Credit History / Age Ratio",
    "person_home_ownership_MORTGAGE": "Home Ownership: Mortgage",
    "person_home_ownership_OTHER": "Home Ownership: Other",
    "person_home_ownership_OWN": "Home Ownership: Own",
    "person_home_ownership_RENT": "Home Ownership: Rent",
    "loan_intent_DEBTCONSOLIDATION": "Loan Intent: Debt Consolidation",
    "loan_intent_EDUCATION": "Loan Intent: Education",
    "loan_intent_HOMEIMPROVEMENT": "Loan Intent: Home Improvement",
    "loan_intent_MEDICAL": "Loan Intent: Medical",
    "loan_intent_PERSONAL": "Loan Intent: Personal",
    "loan_intent_VENTURE": "Loan Intent: Venture",
    "loan_grade_A": "Credit Grade: A (Prime)",
    "loan_grade_B": "Credit Grade: B",
    "loan_grade_C": "Credit Grade: C",
    "loan_grade_D": "Credit Grade: D",
    "loan_grade_E": "Credit Grade: E",
    "loan_grade_F": "Credit Grade: F",
    "loan_grade_G": "Credit Grade: G (Subprime)",
    "cb_person_default_on_file_N": "Historical Default: No",
    "cb_person_default_on_file_Y": "Historical Default: Yes",
}

FIGURES_DIR = ROOT_DIR / "reports" / "figures"
REPORT_CSV_PATH = ROOT_DIR / "reports" / "model_comparison.csv"

# ── Model & Preprocessor Loader ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    preprocessor_path = ROOT_DIR / "models" / "preprocessor.joblib"
    model_path = ROOT_DIR / "models" / "xgboost.joblib"

    if not preprocessor_path.exists():
        st.error(f"Preprocessor not found at: {preprocessor_path}")
        st.stop()
    if not model_path.exists():
        st.error(f"Champion model not found at: {model_path}")
        st.stop()

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    return preprocessor, model, explainer

preprocessor, model, explainer = load_artifacts()

# ── Main Header ──────────────────────────────────────────────────────────────
st.title("💳 Credit Risk Assessment & Underwriting Intelligence")
st.caption(
    "Production-grade decision-support dashboard powered by calibrated machine learning, "
    "neural architectures, and SHAP explainability."
)

# ── Navigation Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🎯 Live Credit Risk Predictor",
    "📈 Model Comparison",
    "🌐 Global Insights & SHAP",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Live Credit Risk Predictor
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📋 Loan Applicant Information")

    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Applicant Demographics")
            person_age = st.number_input("Age (Years)", min_value=18, max_value=100, value=28, step=1)
            person_income = st.number_input("Annual Income ($)", min_value=1000, max_value=2000000, value=65000, step=1000)
            person_emp_length = st.number_input("Employment Length (Years)", min_value=0.0, max_value=60.0, value=4.0, step=0.5)
            person_home_ownership = st.selectbox(
                "Home Ownership",
                options=["RENT", "MORTGAGE", "OWN", "OTHER"],
                index=0,
            )

        with col2:
            st.markdown("##### 💰 Loan Specifications")
            loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=500, max_value=150000, value=12000, step=500)
            loan_int_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=35.0, value=11.5, step=0.1)
            loan_intent = st.selectbox(
                "Loan Purpose / Intent",
                options=["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
                index=0,
            )
            loan_grade = st.selectbox(
                "Assigned Credit Grade",
                options=["A", "B", "C", "D", "E", "F", "G"],
                index=1,
            )

        with col3:
            st.markdown("##### 📜 Credit History & Solvency")
            cb_person_default_on_file = st.selectbox(
                "Historical Default on File?",
                options=["N", "Y"],
                index=0,
            )
            cb_person_cred_hist_length = st.number_input(
                "Credit History Length (Years)",
                min_value=1,
                max_value=40,
                value=6,
                step=1,
            )

            calculated_loan_pct = loan_amnt / max(person_income, 1)
            st.markdown("##### 🔍 Pre-Computed Ratio")
            st.info(f"**Loan as % of Income:** `{calculated_loan_pct * 100:.2f}%`")

        submitted = st.form_submit_button("🚀 Predict Credit Risk", use_container_width=True)

    if submitted:
        loan_percent_income = loan_amnt / max(person_income, 1)

        raw_input_df = pd.DataFrame([{
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
        }])

        try:
            X_transformed = preprocessor.transform(raw_input_df)
        except Exception as e:
            st.error(f"Error during feature preprocessing: {e}")
            st.stop()

        pred_prob = float(model.predict_proba(X_transformed)[:, 1][0])
        pred_pct = pred_prob * 100.0

        if pred_prob < 0.20:
            risk_category = "Low Risk"
            risk_color = "#28a745"
            risk_badge = "🟢 LOW RISK"
            recommendation = (
                "**Recommended Action: Fast-Track Approval.**\n\n"
                "The applicant displays high creditworthiness and low risk of default. "
                "Proceed with standard identity and employment verification without requiring elevated scrutiny."
            )
        elif pred_prob < 0.50:
            risk_category = "Medium Risk"
            risk_color = "#ffc107"
            risk_badge = "🟡 MEDIUM RISK"
            recommendation = (
                "**Recommended Action: Senior Underwriter Review.**\n\n"
                "The applicant carries moderate default risk. A credit officer should perform detailed income "
                "and debt-to-income verification. Consider offering adjusted loan terms, requiring a co-signer, "
                "or securing additional collateral. Automatic rejection is discouraged."
            )
        else:
            risk_category = "High Risk"
            risk_color = "#dc3545"
            risk_badge = "🔴 HIGH RISK"
            recommendation = (
                "**Recommended Action: In-Depth Risk Audit & Mitigation Review.**\n\n"
                "The applicant exhibits elevated default indicators. Underwriting policy mandates human review; "
                "evaluate whether restructured terms (e.g., lower principal, shorter maturity, higher down payment) "
                "or a creditworthy guarantor can bring the profile within acceptable portfolio risk tolerance."
            )

        st.markdown("---")
        st.subheader("🎯 Assessment Results")

        res_col1, res_col2, res_col3 = st.columns([1, 1, 2])

        with res_col1:
            st.metric(
                label="Predicted Default Probability",
                value=f"{pred_pct:.2f}%",
                delta=f"{pred_pct - 21.9:.1f}% vs. Portfolio Avg" if pred_pct != 21.9 else None,
                delta_color="inverse",
            )

        with res_col2:
            st.markdown("**Risk Tier Classification**")
            st.markdown(
                f"<div style='background-color:{risk_color}22; border-left: 6px solid {risk_color}; "
                f"padding: 12px; border-radius: 6px; font-size: 1.25rem; font-weight: bold; color: {risk_color};'>"
                f"{risk_badge}</div>",
                unsafe_allow_html=True,
            )

        with res_col3:
            st.markdown("**Decision-Support Guidance**")
            st.info(recommendation)

        st.markdown("---")
        st.subheader("🔬 Explainability & Risk Attribution (SHAP)")

        shap_explanation = explainer(X_transformed)
        shap_vals = shap_explanation.values[0]
        feature_cols = list(X_transformed.columns)

        friendly_names = [FEATURE_NAME_MAP.get(c, c) for c in feature_cols]
        shap_df = pd.DataFrame({
            "Feature": friendly_names,
            "Raw_Feature": feature_cols,
            "SHAP_Value": shap_vals,
        })

        top_risk_increasing = shap_df[shap_df["SHAP_Value"] > 0].sort_values(by="SHAP_Value", ascending=False).head(3)
        top_risk_reducing = shap_df[shap_df["SHAP_Value"] < 0].sort_values(by="SHAP_Value", ascending=True).head(3)

        factor_col1, factor_col2 = st.columns(2)

        with factor_col1:
            st.markdown("##### 🔺 Top Risk-Increasing Factors (+ SHAP)")
            if not top_risk_increasing.empty:
                for _, row in top_risk_increasing.iterrows():
                    st.markdown(f"- **{row['Feature']}**: `+{row['SHAP_Value']:.3f}` log-odds contribution")
            else:
                st.write("No significant positive risk drivers identified.")

        with factor_col2:
            st.markdown("##### 🔻 Top Risk-Reducing Factors (- SHAP)")
            if not top_risk_reducing.empty:
                for _, row in top_risk_reducing.iterrows():
                    st.markdown(f"- **{row['Feature']}**: `{row['SHAP_Value']:.3f}` log-odds contribution")
            else:
                st.write("No significant protective risk drivers identified.")

        st.markdown("##### 📊 Local SHAP Waterfall Plot")
        st.caption("Illustrates how each feature shifts the applicant's prediction from the base expected value to the final log-odds.")

        shap_single = shap_explanation[0]
        shap_single.feature_names = friendly_names

        # Render compact waterfall plot cleanly via memory buffer to avoid st.pyplot deprecation warnings
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        shap.plots.waterfall(shap_single, max_display=10, show=False)
        plt.title(f"SHAP Waterfall — Applicant Prediction (Risk: {pred_pct:.2f}%)", fontsize=11, fontweight="bold", pad=10)
        plt.tight_layout()

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=150)
        img_buf.seek(0)
        st.image(img_buf, use_container_width=True)
        plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: Model Comparison
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Model Performance Benchmark")
    st.markdown(
        "Comparison of 6 distinct architectures trained with leakage-free preprocessing and stratified cross-validation."
    )

    st.success(
        "🏆 **Champion Model: XGBoost (scale_pos_weight)** — "
        "Achieved highest Test ROC-AUC (**0.9491**), PR-AUC (**0.9076**), and F1-Score (**0.8380**) "
        "with optimal Brier Score probability calibration (**0.0556**)."
    )

    if REPORT_CSV_PATH.exists():
        df_models = pd.read_csv(REPORT_CSV_PATH)

        display_df = df_models.copy()
        # Clean formatting
        for col in ["Test_Accuracy", "Test_Precision", "Test_Recall"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "-")
        for col in ["CV_ROC_AUC", "CV_PR_AUC", "CV_F1", "Test_F1", "Test_ROC_AUC", "Test_PR_AUC", "Test_Brier_Score"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")

        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("Model comparison report not found.")

    st.markdown("---")
    st.subheader("🖼️ Comparative Diagnostic Visualizations")

    mc_col1, mc_col2 = st.columns(2)

    with mc_col1:
        roc_path = FIGURES_DIR / "classical_roc_curves.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC Curves — Classical Baselines vs. Ensembles", use_container_width=True)

        mlp_path = FIGURES_DIR / "mlp_training_curves.png"
        if mlp_path.exists():
            st.image(str(mlp_path), caption="PyTorch MLP — Training & Validation Learning Curves", use_container_width=True)

    with mc_col2:
        cm_path = FIGURES_DIR / "best_classical_confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Champion XGBoost — Test Set Confusion Matrix", use_container_width=True)

        tabnet_path = FIGURES_DIR / "tabnet_feature_importance.png"
        if tabnet_path.exists():
            st.image(str(tabnet_path), caption="TabNet — Attention Mask Feature Importance", use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: Global Insights & SHAP
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🌐 Global Explainability & Risk Factor Insights")
    st.markdown(
        "Examine macroeconomic patterns, credit grade risk gradients, and global SHAP attribution across the entire portfolio."
    )

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        beeswarm_path = FIGURES_DIR / "shap_summary_beeswarm.png"
        if beeswarm_path.exists():
            st.image(
                str(beeswarm_path),
                caption="Global SHAP Beeswarm Summary — Impact Directionality across Top 15 Features",
                use_container_width=True,
            )

        grade_path = FIGURES_DIR / "default_rate_by_grade.png"
        if grade_path.exists():
            st.image(
                str(grade_path),
                caption="Portfolio Default Rate by Assigned Credit Grade (Grade A to G)",
                use_container_width=True,
            )

    with g_col2:
        bar_path = FIGURES_DIR / "shap_summary_bar.png"
        if bar_path.exists():
            st.image(
                str(bar_path),
                caption="Mean |SHAP Value| Global Feature Importance (Top 15 Features)",
                use_container_width=True,
            )

        corr_path = FIGURES_DIR / "correlation_heatmap.png"
        if corr_path.exists():
            st.image(
                str(corr_path),
                caption="Feature Correlation Heatmap (Base & Engineered Banking Ratios)",
                use_container_width=True,
            )
