import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Plot dyadic representation alignment metrics.")
    parser.add_argument("--detailed_csv", type=str, required=True, help="Path to dyadic_per_pair_metrics.csv")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.detailed_csv)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    space_palette = {"Absolute": "#34495e", "Relative": "#e74c3c"}

    # Filter for Real Dyads for main metric curves
    df_real = df[df["dyad_type"] == "Real"].copy()

    # --- FIGURE 1: 4-PANEL DYADIC ALIGNMENT PROFILE ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: Open Condition Centroid Cosine Similarity
    sns.lineplot(
        data=df_real, x="layer", y="centroid_cosine_open", hue="space", style="space",
        markers=True, dashes=False, markersize=8, linewidth=2.5, palette=space_palette,
        ax=axes[0, 0], errorbar="se"
    )
    axes[0, 0].set_title("A. Dyad Inter-Brain Centroid Similarity (Open / Interactive)", weight="bold", pad=10)
    axes[0, 0].set_xlabel("LaBraM Layer Index")
    axes[0, 0].set_ylabel("Cosine Similarity (Mean ± SE)")
    axes[0, 0].set_xticks(range(12))

    # Panel B: Zero-Shot Cross-Subject Probe Transfer
    sns.lineplot(
        data=df_real, x="layer", y="probe_cross_acc", hue="space", style="space",
        markers=True, dashes=False, markersize=8, linewidth=2.5, palette=space_palette,
        ax=axes[0, 1], errorbar="se"
    )
    axes[0, 1].axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Chance Level (50%)")
    axes[0, 1].set_title("B. Zero-Shot Cross-Subject Probe Transfer (P1 ↔ P2)", weight="bold", pad=10)
    axes[0, 1].set_xlabel("LaBraM Layer Index")
    axes[0, 1].set_ylabel("Balanced Accuracy (Mean ± SE)")
    axes[0, 1].set_ylim([0.45, 0.95])
    axes[0, 1].set_xticks(range(12))

    # Panel C: Distribution MMD Divergence (Lower is better)
    sns.lineplot(
        data=df_real, x="layer", y="mmd_overall", hue="space", style="space",
        markers=True, dashes=False, markersize=8, linewidth=2.5, palette=space_palette,
        ax=axes[1, 0], errorbar="se"
    )
    axes[1, 0].set_title("C. Inter-Subject Distribution Divergence (RBF MMD² ↓)", weight="bold", pad=10)
    axes[1, 0].set_xlabel("LaBraM Layer Index")
    axes[1, 0].set_ylabel("MMD² Distance (Mean ± SE)")
    axes[1, 0].set_xticks(range(12))

    # Panel D: Open vs. Closed Contrast within Dyads
    df_real["delta_open_closed"] = df_real["centroid_cosine_open"] - df_real["centroid_cosine_closed"]
    sns.lineplot(
        data=df_real, x="layer", y="delta_open_closed", hue="space", style="space",
        markers=True, dashes=False, markersize=8, linewidth=2.5, palette=space_palette,
        ax=axes[1, 1], errorbar="se"
    )
    axes[1, 1].axhline(0.0, color="black", linestyle=":", alpha=0.6)
    axes[1, 1].set_title("D. Social Specificity: (Cosine Open - Cosine Closed)", weight="bold", pad=10)
    axes[1, 1].set_xlabel("LaBraM Layer Index")
    axes[1, 1].set_ylabel("Δ Cosine (Mean ± SE)")
    axes[1, 1].set_xticks(range(12))

    plt.tight_layout()
    fig1_path = os.path.join(args.output_dir, "dyadic_representation_alignment.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved primary 4-panel alignment plot to: {fig1_path}")

    # --- FIGURE 2: REAL VS. SURROGATE DYAD GAP ---
    # Reshape to compute real - surrogate contrast
    piv = df.pivot_table(index=["layer", "space", "pair_id"], columns="dyad_type", values="centroid_cosine_open").reset_index()
    if "Surrogate" in piv.columns and "Real" in piv.columns:
        piv["dyadic_gap"] = piv["Real"] - piv["Surrogate"]
        plt.figure(figsize=(10, 5))
        sns.lineplot(
            data=piv, x="layer", y="dyadic_gap", hue="space", style="space",
            markers=True, dashes=False, markersize=9, linewidth=2.5, palette=space_palette,
            errorbar="se"
        )
        plt.axhline(0.0, color="black", linestyle="--", alpha=0.6)
        plt.title("Interpersonal Coupling: Real vs. Surrogate Dyad Contrast (Open Screen)", weight="bold", pad=12)
        plt.xlabel("LaBraM Layer Index")
        plt.ylabel("Δ Cosine (Real - Surrogate)")
        plt.xticks(range(12))
        plt.tight_layout()
        fig2_path = os.path.join(args.output_dir, "dyad_real_vs_surrogate_gap.png")
        plt.savefig(fig2_path, dpi=300)
        plt.close()
        print(f"Saved Real vs. Surrogate gap plot to: {fig2_path}")

if __name__ == "__main__":
    main()