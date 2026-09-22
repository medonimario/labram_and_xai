import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def sample_balanced_subset(labels, samples_per_class=400, seed=42):
    """Samples an equal number of points from each class to avoid overplotting."""
    np.random.seed(seed)
    unique_labels = np.unique(labels)
    selected_indices = []

    for lbl in unique_labels:
        cls_indices = np.where(labels == lbl)[0]
        n_select = min(len(cls_indices), samples_per_class)
        chosen = np.random.choice(cls_indices, size=n_select, replace=False)
        selected_indices.extend(chosen)

    return np.array(selected_indices)

def plot_layer_projections(
    layer_id,
    abs_a, abs_b,
    rel_a, rel_b,
    sub_indices,
    labels,
    output_dir,
    method="pca",
    seed=42
):
    """
    Generates a 2x2 qualitative plot comparing Model A and Model B 
    in Absolute (top row) vs Relative (bottom row) spaces.
    """
    sub_labels = labels[sub_indices]
    
    # Slice the sampled data
    X_abs_a = abs_a[sub_indices]
    X_abs_b = abs_b[sub_indices]
    X_rel_a = rel_a[sub_indices]
    X_rel_b = rel_b[sub_indices]

    def reduce_dim(data_a, data_b, mode):
        if mode == "pca":
            # Fit PCA independently on each space (Moschella et al. Figure 5/11)
            pca_a = PCA(n_components=2, random_state=seed)
            pca_b = PCA(n_components=2, random_state=seed)
            return pca_a.fit_transform(data_a), pca_b.fit_transform(data_b)
        elif mode == "tsne":
            tsne_a = TSNE(n_components=2, perplexity=30, random_state=seed)
            tsne_b = TSNE(n_components=2, perplexity=30, random_state=seed)
            return tsne_a.fit_transform(data_a), tsne_b.fit_transform(data_b)
        else:
            raise ValueError(f"Unknown reduction method: {mode}")

    abs_a_2d, abs_b_2d = reduce_dim(X_abs_a, X_abs_b, method)
    rel_a_2d, rel_b_2d = reduce_dim(X_rel_a, X_rel_b, method)

    # Color mapping: 0 = Closed (blue), 1 = Open (orange/red)
    color_map = {0: "#1f77b4", 1: "#d62728"}
    colors = [color_map.get(lbl, "#7f7f7f") for lbl in sub_labels]
    class_names = {0: "Closed Screen", 1: "Open Screen (Interactive)"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    configs = [
        (axes[0, 0], abs_a_2d, "Absolute - Model A (Fold i)"),
        (axes[0, 1], abs_b_2d, "Absolute - Model B (Fold j)"),
        (axes[1, 0], rel_a_2d, "Relative - Model A (Fold i)"),
        (axes[1, 1], rel_b_2d, "Relative - Model B (Fold j)")
    ]

    for ax, data_2d, title in configs:
        for lbl_val, lbl_name in class_names.items():
            mask = (sub_labels == lbl_val)
            ax.scatter(
                data_2d[mask, 0],
                data_2d[mask, 1],
                c=color_map[lbl_val],
                label=lbl_name,
                alpha=0.6,
                edgecolors="none",
                s=28
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    axes[0, 0].set_ylabel("Absolute", fontsize=14, fontweight="bold")
    axes[1, 0].set_ylabel("Relative", fontsize=14, fontweight="bold")
    axes[0, 1].legend(loc="upper right", frameon=True, fontsize=10)

    plt.suptitle(f"Layer {layer_id} Representation Geometry ({method.upper()})", fontsize=15, y=0.98)
    plt.tight_layout()

    out_file = os.path.join(output_dir, f"layer_{layer_id}_{method}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate qualitative 2D latent space visualizations.")
    parser.add_argument("--model_a_proj", type=str, required=True, help="Path to model_a_relative.pkl")
    parser.add_argument("--model_b_proj", type=str, required=True, help="Path to model_b_relative.pkl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--samples_per_class", type=int, default=400, help="Number of samples per condition to plot")
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "tsne"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.model_a_proj, "rb") as f:
        proj_a = pickle.load(f)
    with open(args.model_b_proj, "rb") as f:
        proj_b = pickle.load(f)

    labels = np.array(proj_a["eval_meta"]["y"])
    layers = sorted(proj_a["relative_activations"].keys())

    # Sample a fixed balanced subset so all layer figures compare identical physical trials
    sub_indices = sample_balanced_subset(labels, samples_per_class=args.samples_per_class, seed=args.seed)
    print(f"Selected {len(sub_indices)} balanced samples across classes for 2D visualization.")

    for l_id in layers:
        print(f"Generating 2D {args.method.upper()} plot for Layer {l_id}...")
        abs_a = proj_a["abs_norm_activations"][l_id]
        abs_b = proj_b["abs_norm_activations"][l_id]
        rel_a = proj_a["relative_activations"][l_id]
        rel_b = proj_b["relative_activations"][l_id]

        plot_layer_projections(
            layer_id=l_id,
            abs_a=abs_a,
            abs_b=abs_b,
            rel_a=rel_a,
            rel_b=rel_b,
            sub_indices=sub_indices,
            labels=labels,
            output_dir=args.output_dir,
            method=args.method,
            seed=args.seed
        )

    print(f"All qualitative plots saved in: {args.output_dir}")

if __name__ == "__main__":
    main()