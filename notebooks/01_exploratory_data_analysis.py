import os

import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import download_dataset


def run_eda(output_dir: str = "reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    df = download_dataset()

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Target Class Balance
    plt.figure(figsize=(7, 5))
    sns.countplot(
        x="target",
        data=df,
        hue="target",
        palette=["#2ecc71", "#e74c3c"],
        legend=False,
    )
    plt.title("Target Distribution (0 = Healthy, 1 = Disease Risk)", pad=15)
    plt.xlabel("Diagnostic Classification")
    plt.ylabel("Patient Count")
    plt.xticks([0, 1], ["Healthy (0)", "Disease Risk (1)"])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_balance.png", dpi=300)
    plt.close()

    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 9))
    corr = df.corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.75,
    )
    plt.title("Correlation Matrix of Clinical Indicators", pad=15)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300)
    plt.close()

    # 3. Clinical Feature Histograms
    features_to_plot = ["age", "trestbps", "chol", "thalach"]
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, col in enumerate(features_to_plot):
        r, c = divmod(i, 2)
        sns.histplot(
            data=df,
            x=col,
            hue="target",
            kde=True,
            palette=["#2ecc71", "#e74c3c"],
            ax=axes[r, c],
            bins=20,
        )
        axes[r, c].set_title(f"Distribution of '{col}'")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distributions.png", dpi=300)
    plt.close()
    print(f"Figures saved successfully in {output_dir}/")


if __name__ == "__main__":
    run_eda()