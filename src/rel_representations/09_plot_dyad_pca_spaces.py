import os
import re
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict
from itertools import combinations

def parse_metadata_from_path(filepath):
    """
    Extracts pair_id, subject_id, condition, and segment_idx from filename.
    """
    fname = os.path.basename(filepath)
    
    pair_match = re.search(r'pair(\d+)', fname, re.IGNORECASE)
    pair_id = pair_match.group(1) if pair_match else "unknown"

    cond_match = re.search(r'condition_([a-zA-Z0-9]+)', fname)
    condition = cond_match.group(1) if cond_match else "unknown"

    seg_match = re.search(r'segment(\d+)', fname)
    seg_idx = int(seg_match.group(1)) if seg_match else -1

    sub_match = re.search(r'pair\d+[_-]?([a-zA-Z0-9]+?)(?:_condition|_ave|\.pkl)', fname, re.IGNORECASE)
    if sub_match and sub_match.group(1).strip('_'):
        subject_id = sub_match.group(1).strip('_')
    else:
        fallback = re.search(r'([pP]\d+|sub(?:ject)?\d+|_\d+_)', fname)
        subject_id = fallback.group(1).strip('_') if fallback else "p1"

    return {
        "filepath": filepath,
        "pair_id": pair_id,
        "subject_id": subject_id,
        "condition": condition,
        "segment_idx": seg_idx
    }

def get_balanced_indices(labels, max_per_class=200, seed=42):
    """Samples up to max_per_class examples per condition to prevent visual clutter."""
    np.random.seed(seed)
    chosen_indices = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        n_take = min(len(cls_idx), max_per_class)
        chosen = np.random.choice(cls_idx, size=n_take, replace=False)
        chosen_indices.extend(chosen)
    return np.array(sorted(chosen_indices))

def plot_dyad_layer(
    layer_id,
    pair_id,
    sub1_id,
    sub2_id,
    abs_1, y_1,
    abs_2, y_2,
    rel_1,
    rel_2,
    output_dir,
    method="pca",
    include_overlay=False,
    seed=42
):
    """
    Renders 2D latent projections for a single dyad at a specific layer.
    """
    def reduce_dimension(data_1, data_2, mode):
        if mode == "pca":
            # Independent PCAs reveal internal manifold geometry (Moschella et al. Fig 5)
            pca1 = PCA(n_components=2, random_state=seed)
            pca2 = PCA(n_components=2, random_state=seed)
            p1_2d = pca1.fit_transform(data_1)
            p2_2d = pca2.fit_transform(data_2)

            # Joint PCA for spatial co-registration / overlay
            pca_joint = PCA(n_components=2, random_state=seed)
            all_joint = pca_joint.fit_transform(np.vstack([data_1, data_2]))
            joint_1 = all_joint[:len(data_1)]
            joint_2 = all_joint[len(data_1):]
            return p1_2d, p2_2d, joint_1, joint_2
        elif mode == "tsne":
            tsne1 = TSNE(n_components=2, perplexity=20, random_state=seed)
            tsne2 = TSNE(n_components=2, perplexity=20, random_state=seed)
            p1_2d = tsne1.fit_transform(data_1)
            p2_2d = tsne2.fit_transform(data_2)

            tsne_joint = TSNE(n_components=2, perplexity=20, random_state=seed)
            all_joint = tsne_joint.fit_transform(np.vstack([data_1, data_2]))
            joint_1 = all_joint[:len(data_1)]
            joint_2 = all_joint[len(data_1):]
            return p1_2d, p2_2d, joint_1, joint_2
        else:
            raise ValueError(f"Unknown reduction method: {mode}")

    # Compute 2D coordinates
    abs1_2d, abs2_2d, abs_joint1, abs_joint2 = reduce_dimension(abs_1, abs_2, method)
    rel1_2d, rel2_2d, rel_joint1, rel_joint2 = reduce_dimension(rel_1, rel_2, method)

    # Styling definitions
    color_map = {0: "#1f77b4", 1: "#d62728"}
    class_names = {0: "Closed Screen", 1: "Open Screen"}

    ncols = 3 if include_overlay else 2
    figsize = (15, 9) if include_overlay else (10.5, 9)
    fig, axes = plt.subplots(2, ncols, figsize=figsize)

    # Helper function to plot individual points
    def scatter_sub(ax, data_2d, labels, title):
        for val, name in class_names.items():
            mask = (labels == val)
            if np.any(mask):
                ax.scatter(
                    data_2d[mask, 0], data_2d[mask, 1],
                    c=color_map[val], label=name,
                    alpha=0.65, edgecolors="none", s=32
                )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Helper function to plot joint overlay
    def scatter_joint(ax, p1_data, p2_data, labels1, labels2, title):
        for val, name in class_names.items():
            # Subject 1: Circles
            mask1 = (labels1 == val)
            if np.any(mask1):
                ax.scatter(
                    p1_data[mask1, 0], p1_data[mask1, 1],
                    c=color_map[val], marker='o', alpha=0.6,
                    edgecolors="none", s=32, label=f"P1 ({name})"
                )
            # Subject 2: Triangles
            mask2 = (labels2 == val)
            if np.any(mask2):
                ax.scatter(
                    p2_data[mask2, 0], p2_data[mask2, 1],
                    c=color_map[val], marker='^', alpha=0.6,
                    edgecolors="none", s=36, label=f"P2 ({name})"
                )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # --- ROW 0: ABSOLUTE SPACE ---
    scatter_sub(axes[0, 0], abs1_2d, y_1, f"Absolute — {sub1_id}")
    scatter_sub(axes[0, 1], abs2_2d, y_2, f"Absolute — {sub2_id}")
    if include_overlay:
        scatter_joint(axes[0, 2], abs_joint1, abs_joint2, y_1, y_2, f"Absolute — Joint Overlay")

    # --- ROW 1: RELATIVE SPACE ---
    scatter_sub(axes[1, 0], rel1_2d, y_1, f"Relative — {sub1_id}")
    scatter_sub(axes[1, 1], rel2_2d, y_2, f"Relative — {sub2_id}")
    if include_overlay:
        scatter_joint(axes[1, 2], rel_joint1, rel_joint2, y_1, y_2, f"Relative — Joint Overlay")

    # Labels and Legend
    axes[0, 0].set_ylabel("Absolute Space", fontsize=14, fontweight="bold")
    axes[1, 0].set_ylabel("Relative Space", fontsize=14, fontweight="bold")
    
    if include_overlay:
        axes[0, 2].legend(loc="upper right", frameon=True, fontsize=8, ncol=2)
    else:
        axes[0, 1].legend(loc="upper right", frameon=True, fontsize=9)

    plt.suptitle(f"Pair {pair_id} ({sub1_id} vs {sub2_id}) — Layer {layer_id} {method.upper()}", fontsize=15, y=0.98)
    plt.tight_layout()

    out_filename = f"layer_{layer_id}_{method}.png"
    out_path = os.path.join(output_dir, out_filename)
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate qualitative 2D plots for dyad pairs across layers.")
    parser.add_argument("--model_proj", type=str, required=True, help="Path to model_a_relative.pkl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save pair folders")
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "tsne"])
    parser.add_argument("--samples_per_class", type=int, default=200, help="Max points per condition to display")
    parser.add_argument("--include_overlay", action="store_true", help="Include joint overlay as 3rd column (2x3 grid)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading representations from: {args.model_proj}")
    with open(args.model_proj, "rb") as f:
        proj_data = pickle.load(f)

    paths = proj_data["eval_meta"]["paths"]
    labels = np.array(proj_data["eval_meta"]["y"])
    layers = sorted(proj_data["relative_activations"].keys())

    # 1. Parse metadata and group by dyad
    metadata = [parse_metadata_from_path(p) for p in paths]
    dyads = defaultdict(lambda: defaultdict(list))
    for i, meta in enumerate(metadata):
        dyads[meta["pair_id"]][meta["subject_id"]].append(i)

    # 2. Filter for valid pairs with at least 2 participants (>= 10 samples each)
    valid_dyads = {}
    for p_id, sub_dict in dyads.items():
        qualifying = {s: idxs for s, idxs in sub_dict.items() if len(idxs) >= 10}
        if len(qualifying) >= 2:
            valid_dyads[p_id] = qualifying

    print(f"Found {len(valid_dyads)} valid pairs for qualitative visualization: {list(valid_dyads.keys())}")

    # 3. Iterate over each dyad
    for p_id, sub_dict in valid_dyads.items():
        subs = sorted(sub_dict.keys())
        # Form primary dyad pair (e.g., P1 vs P2)
        s1, s2 = subs[0], subs[1]

        # Create separate output folder for each pair
        pair_folder = os.path.join(args.output_dir, f"pair_{p_id}")
        os.makedirs(pair_folder, exist_ok=True)
        print(f"\nProcessing Pair {p_id} ({s1} vs {s2}) -> {pair_folder}")

        # Extract subject indices and sample balanced subsets
        raw_idx1 = np.array(sub_dict[s1])
        raw_idx2 = np.array(sub_dict[s2])

        sub_sel1 = get_balanced_indices(labels[raw_idx1], max_per_class=args.samples_per_class, seed=args.seed)
        sub_sel2 = get_balanced_indices(labels[raw_idx2], max_per_class=args.samples_per_class, seed=args.seed)

        idx1 = raw_idx1[sub_sel1]
        idx2 = raw_idx2[sub_sel2]

        y1 = labels[idx1]
        y2 = labels[idx2]

        for l_id in layers:
            abs_all = proj_data["abs_norm_activations"][l_id]
            rel_all = proj_data["relative_activations"][l_id]

            abs_1 = abs_all[idx1]
            abs_2 = abs_all[idx2]
            rel_1 = rel_all[idx1]
            rel_2 = rel_all[idx2]

            plot_dyad_layer(
                layer_id=l_id,
                pair_id=p_id,
                sub1_id=s1,
                sub2_id=s2,
                abs_1=abs_1, y_1=y1,
                abs_2=abs_2, y_2=y2,
                rel_1=rel_1,
                rel_2=rel_2,
                output_dir=pair_folder,
                method=args.method,
                include_overlay=args.include_overlay,
                seed=args.seed
            )

    print(f"\nAll qualitative dyad figures generated successfully in: {args.output_dir}")

if __name__ == "__main__":
    main()