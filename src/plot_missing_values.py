from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "credit_risk_dataset.csv"
PROCESSED_TRAIN_PATH = ROOT_DIR / "data" / "processed" / "train.csv"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OUT_BEFORE_PATH = FIGURES_DIR / "missing_values_before_preprocessing.png"
OUT_AFTER_PATH = FIGURES_DIR / "missing_values_after_preprocessing.png"

# Purple colormap: soft lavender for present (0), deep royal purple for missing (1)
PURPLE_PRESENT = "#F3E8FF"  # Soft lavender
PURPLE_MISSING = "#6B21A8"  # Deep royal purple
PURPLE_CMAP = ListedColormap([PURPLE_PRESENT, PURPLE_MISSING])


def plot_missing_heatmap(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    output_path: Path,
    figsize: tuple = (10, 6),
    rotation: int = 45,
) -> None:
    """Generate and save a publication-quality purple missing-value heatmap."""
    fig, ax = plt.subplots(figsize=figsize)

    # Missing value boolean matrix
    missing_matrix = df.isnull()

    sns.heatmap(
        missing_matrix,
        cmap=PURPLE_CMAP,
        vmin=0,
        vmax=1,
        cbar=False,
        yticklabels=False,
        ax=ax,
    )

    # Clean styling
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    plt.suptitle(subtitle, fontsize=10, color="#4B5563", y=0.92)
    ax.set_xlabel("Columns / Features", fontsize=11, fontweight="medium", labelpad=10)
    ax.set_ylabel("Records (Observations)", fontsize=11, fontweight="medium", labelpad=10)
    plt.xticks(rotation=rotation, ha="right", fontsize=9)

    # Custom legend for binary missingness
    legend_patches = [
        mpatches.Patch(facecolor=PURPLE_PRESENT, edgecolor="#9333EA", label="Present (Non-Null)"),
        mpatches.Patch(facecolor=PURPLE_MISSING, edgecolor="#3B0764", label="Missing (NaN)"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.12),
        frameon=True,
        facecolor="white",
        edgecolor="#D1D5DB",
        fontsize=9,
        ncol=2,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    print("Loading datasets...")
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_train = pd.read_csv(PROCESSED_TRAIN_PATH)

    raw_missing_total = df_raw.isnull().sum().sum()
    raw_missing_cols = df_raw.isnull().sum()
    raw_missing_cols = raw_missing_cols[raw_missing_cols > 0].to_dict()
    print(f"Raw dataset shape: {df_raw.shape} | Total missing cells: {raw_missing_total:,}")
    print(f"Missing columns in raw data: {raw_missing_cols}")

    train_missing_total = df_train.isnull().sum().sum()
    print(f"Processed train dataset shape: {df_train.shape} | Total missing cells: {train_missing_total:,}")

    # 1. Before Preprocessing Heatmap
    subtitle_before = (
        f"Raw Data: {len(df_raw):,} records × {df_raw.shape[1]} columns | "
        f"Missing cells: {raw_missing_total:,} (person_emp_length: {raw_missing_cols.get('person_emp_length', 0):,}, "
        f"loan_int_rate: {raw_missing_cols.get('loan_int_rate', 0):,})"
    )
    plot_missing_heatmap(
        df=df_raw,
        title="Missing Values Heatmap — Before Preprocessing (Raw Data)",
        subtitle=subtitle_before,
        output_path=OUT_BEFORE_PATH,
        figsize=(11, 6),
        rotation=35,
    )

    # 2. After Preprocessing Heatmap
    subtitle_after = (
        f"Processed Train Data: {len(df_train):,} records × {df_train.shape[1]} columns | "
        f"Missing cells: {train_missing_total} (Zero Data Leakage Median Imputation Complete)"
    )
    plot_missing_heatmap(
        df=df_train,
        title="Missing Values Heatmap — After Preprocessing (Training Set)",
        subtitle=subtitle_after,
        output_path=OUT_AFTER_PATH,
        figsize=(14, 6.5),
        rotation=55,
    )

    print("\nMissing-value heatmaps generated successfully!")


if __name__ == "__main__":
    main()
