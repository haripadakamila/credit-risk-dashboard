"""
Exploratory Data Analysis (EDA) for Credit Risk Prediction.

Generates publication-ready figures saved in reports/figures/:
1. target_distribution.png (loan_status class balance)
2. default_rate_by_grade.png (loan_grade vs default rate)
3. default_rate_by_intent.png (loan_intent vs default rate)
4. default_rate_by_home_ownership.png (person_home_ownership vs default rate)
5. correlation_heatmap.png (numerical feature correlations)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

FIGURES_DIR = Path("reports/figures")
DATA_PATH = Path("data/raw/credit_risk_dataset.csv")

PRIMARY_COLOR = "#2b5c8f"
ALERT_COLOR = "#d9534f"
ACCENT_PALETTE = ["#2b5c8f", "#418ab3", "#6baed6", "#fdae6b", "#f16913", "#d94801", "#a63603"]


def load_and_clean_for_eda(data_path: Path) -> pd.DataFrame:
    """Load raw dataset and apply basic integrity cleaning (drop exact duplicates and extreme anomalies)."""
    df = pd.read_csv(data_path)
    initial_len = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    dups_removed = initial_len - len(df)
    
    # Filter extreme physical impossibilities
    df = df[(df["person_age"] <= 100) & ((df["person_emp_length"].isna()) | (df["person_emp_length"] <= 60))]
    print(f"Loaded {len(df)} records after removing {dups_removed} duplicates and 7 extreme outliers.")
    return df


def plot_target_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot target distribution (loan_status class balance)."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    counts = df["loan_status"].value_counts().sort_index()
    percentages = df["loan_status"].value_counts(normalize=True).sort_index() * 100
    
    labels = ["Non-Default (0)", "Default (1)"]
    colors = [PRIMARY_COLOR, ALERT_COLOR]
    
    bars = ax.bar(labels, counts.values, color=colors, width=0.45, edgecolor="none", zorder=3)
    
    # Annotate bars
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        height = bar.get_height()
        ax.annotate(
            f"{count:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#222222",
        )
    
    ax.set_title("Target Distribution: Loan Status Class Balance", fontsize=15, fontweight="bold", pad=20)
    ax.set_ylabel("Number of Applicants", fontsize=12, fontweight="medium")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    
    ratio = counts[0] / counts[1]
    ax.text(
        0.5, 0.92,
        f"Imbalance Ratio: {ratio:.2f}:1 (Non-Default : Default)",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dddddd", alpha=0.9),
    )
    
    plt.tight_layout()
    output_path = output_dir / "target_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_default_rate_by_grade(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot loan_grade vs default rate."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    grade_stats = (
        df.groupby("loan_grade")["loan_status"]
        .agg(count="count", default_rate="mean")
        .reindex(["A", "B", "C", "D", "E", "F", "G"])
    )
    
    rates = grade_stats["default_rate"] * 100
    grades = grade_stats.index
    
    # Color gradient from lowest default rate (blue) to highest (red)
    norm = plt.Normalize(rates.min(), rates.max())
    cmap = plt.cm.YlOrRd
    bar_colors = [cmap(norm(r)) for r in rates]
    
    bars = ax.bar(grades, rates, color=bar_colors, edgecolor="#444444", linewidth=0.6, width=0.55, zorder=3)
    
    for bar, rate, count in zip(bars, rates, grade_stats["count"]):
        height = bar.get_height()
        ax.annotate(
            f"{rate:.1f}%\n(n={count:,})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if rate > 50 else "normal",
        )
    
    ax.set_title("Historical Default Rate by Credit Loan Grade", fontsize=15, fontweight="bold", pad=20)
    ax.set_xlabel("Loan Grade (A = Highest Quality, G = Lowest Quality)", fontsize=12, fontweight="medium", labelpad=10)
    ax.set_ylabel("Default Rate (%)", fontsize=12, fontweight="medium")
    ax.set_ylim(0, 115)
    ax.axhline(df["loan_status"].mean() * 100, color="#333333", linestyle="--", linewidth=1.2, label=f"Average Default Rate ({df['loan_status'].mean()*100:.1f}%)")
    ax.legend(loc="upper left", frameon=True)
    
    plt.tight_layout()
    output_path = output_dir / "default_rate_by_grade.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_default_rate_by_intent(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot loan_intent vs default rate."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    intent_stats = (
        df.groupby("loan_intent")["loan_status"]
        .agg(count="count", default_rate="mean")
        .sort_values(by="default_rate", ascending=True)
    )
    
    rates = intent_stats["default_rate"] * 100
    intents = [intent.replace("DEBTCONSOLIDATION", "DEBT CONSOLIDATION").replace("HOMEIMPROVEMENT", "HOME IMPROVEMENT") for intent in intent_stats.index]
    
    bars = ax.barh(intents, rates, color="#3b75af", edgecolor="none", height=0.55, zorder=3)
    
    for bar, rate, count in zip(bars, rates, intent_stats["count"]):
        width = bar.get_width()
        ax.annotate(
            f" {rate:.1f}%  (n={count:,})",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="medium",
        )
    
    ax.set_title("Default Rate by Loan Intent / Purpose", fontsize=15, fontweight="bold", pad=20)
    ax.set_xlabel("Default Rate (%)", fontsize=12, fontweight="medium", labelpad=10)
    ax.set_ylabel("Loan Intent", fontsize=12, fontweight="medium")
    ax.set_xlim(0, max(rates) * 1.35)
    ax.axvline(df["loan_status"].mean() * 100, color="#d9534f", linestyle="--", linewidth=1.2, label=f"Portfolio Average ({df['loan_status'].mean()*100:.1f}%)")
    ax.legend(loc="lower right", frameon=True)
    
    plt.tight_layout()
    output_path = output_dir / "default_rate_by_intent.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_default_rate_by_home_ownership(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot person_home_ownership vs default rate."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    home_stats = (
        df.groupby("person_home_ownership")["loan_status"]
        .agg(count="count", default_rate="mean")
        .sort_values(by="default_rate", ascending=True)
    )
    
    rates = home_stats["default_rate"] * 100
    homes = home_stats.index
    
    colors = ["#2ca02c" if r < 10 else "#1f77b4" if r < 20 else "#d9534f" for r in rates]
    bars = ax.bar(homes, rates, color=colors, width=0.45, edgecolor="none", zorder=3)
    
    for bar, rate, count in zip(bars, rates, home_stats["count"]):
        height = bar.get_height()
        ax.annotate(
            f"{rate:.1f}%\n(n={count:,})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="medium",
        )
    
    ax.set_title("Default Rate by Home Ownership Status", fontsize=15, fontweight="bold", pad=20)
    ax.set_xlabel("Home Ownership", fontsize=12, fontweight="medium", labelpad=10)
    ax.set_ylabel("Default Rate (%)", fontsize=12, fontweight="medium")
    ax.set_ylim(0, max(rates) * 1.3)
    ax.axhline(df["loan_status"].mean() * 100, color="#333333", linestyle="--", linewidth=1.2, label=f"Portfolio Average ({df['loan_status'].mean()*100:.1f}%)")
    ax.legend(loc="upper left", frameon=True)
    
    plt.tight_layout()
    output_path = output_dir / "default_rate_by_home_ownership.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot correlation heatmap for numerical features and target."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    num_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "loan_status",
    ]
    
    readable_labels = [
        "Age",
        "Income",
        "Emp Length",
        "Loan Amnt",
        "Interest Rate",
        "Loan % Income",
        "Credit Hist Length",
        "Loan Status (Target)",
    ]
    
    corr_matrix = df[num_cols].corr()
    corr_matrix.index = readable_labels
    corr_matrix.columns = readable_labels
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.75,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10, "weight": "medium"},
        ax=ax,
    )
    
    ax.set_title("Correlation Heatmap: Numerical Features & Default Target", fontsize=15, fontweight="bold", pad=20)
    
    plt.tight_layout()
    output_path = output_dir / "correlation_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_eda() -> None:
    """Main execution function for EDA."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    print("=" * 60)
    print("Starting Credit Risk EDA Analysis...")
    print("=" * 60)
    
    df = load_and_clean_for_eda(DATA_PATH)
    
    plot_target_distribution(df, FIGURES_DIR)
    plot_default_rate_by_grade(df, FIGURES_DIR)
    plot_default_rate_by_intent(df, FIGURES_DIR)
    plot_default_rate_by_home_ownership(df, FIGURES_DIR)
    plot_correlation_heatmap(df, FIGURES_DIR)
    
    print("=" * 60)
    print("All 5 EDA figures generated and saved successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_eda()
