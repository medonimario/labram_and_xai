import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to alignment_metrics_summary.csv")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.csv_path)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = {"Absolute": "#34495e", "Relative": "#e74c3c"}

    metrics = [
        ("cosine_mean", "Pairwise Cosine Similarity (↑)", [0.0, 1.05]),
        ("mrr", "Mean Reciprocal Rank (MRR @ All) (↑)", [0.0, 1.05]),
        ("jaccard", "Neighborhood Jaccard Similarity (k=10) (↑)", [0.0, 1.05])
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    for ax, (metric_col, title, ylim) in zip(axes, metrics):
        sns.lineplot(
            data=df,
            x="layer",
            y=metric_col,
            hue="space",
            style="space",
            markers=True,
            dashes=False,
            markersize=9,
            linewidth=2.5,
            palette=palette,
            ax=ax
        )
        ax.set_title(title, weight="bold", pad=10)
        ax.set_xlabel("LaBraM Transformer Layer Index")
        ax.set_ylabel("Metric Score")
        ax.set_ylim(ylim)
        ax.set_xticks(range(12))
        ax.legend(title="Representation", loc="upper left")

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "alignment_metrics_across_layers.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved alignment trajectory plots to: {plot_path}")

    # Condition breakdown plot (Open vs Closed)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for ax, cond in zip(axes, ["closed", "open"]):
        sns.lineplot(
            data=df,
            x="layer",
            y=f"cosine_{cond}",
            hue="space",
            style="space",
            markers=True,
            dashes=False,
            markersize=8,
            linewidth=2.2,
            palette=palette,
            ax=ax
        )
        ax.set_title(f"Cosine Similarity ({cond.capitalize()} Condition)", weight="bold")
        ax.set_xlabel("LaBraM Transformer Layer Index")
        ax.set_ylabel("Cosine Similarity")
        ax.set_ylim([0.0, 1.05])
        ax.set_xticks(range(12))

    plt.tight_layout()
    cond_plot_path = os.path.join(args.output_dir, "alignment_by_condition.png")
    plt.savefig(cond_plot_path, dpi=300)
    plt.close()
    print(f"Saved condition-specific breakdown to: {cond_plot_path}")

if __name__ == "__main__":
    main()