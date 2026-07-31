import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import mne
from mne.viz.topomap import _find_topomap_coords

def get_channel_coords(ch_names):
    """
    Returns 2D coordinates for the dataset channel list using MNE's biosemi64 montage.
    Automatically handles capitalization differences (e.g., 'FPZ' -> 'Fpz').
    """
    # Fix casing to match standard biosemi64 naming
    mne_names = [ch.replace('Z', 'z').replace('FP', 'Fp') for ch in ch_names]
    
    # Use the biosemi64 montage for an even circular distribution
    montage = mne.channels.make_standard_montage('biosemi64')
    
    try:
        info = mne.create_info(ch_names=mne_names, sfreq=1, ch_types='eeg')
        info.set_montage(montage)
        
        # Project 3D positions to 2D
        pos_3d = np.array([montage.get_positions()['ch_pos'][ch] for ch in mne_names])
        max_dist = np.max(np.linalg.norm(pos_3d, axis=1))
        
        # Create a sphere slightly larger than the head for projection
        custom_sphere = (0.0, 0.0, 0.0, max_dist)
        picks = range(len(mne_names))
        pos_2d = _find_topomap_coords(info, picks=picks, sphere=custom_sphere)
        
        return pos_2d, max_dist
        
    except ValueError as e:
        print(f"Error creating montage: {e}")
        print("Falling back to random coordinates. Check channel names.")
        return np.random.rand(len(ch_names), 2), 1.0


def plot_attention_topomaps(attn_data, ch_names, coords, max_dist, title, save_path, cmap_type="absolute"):
    """
    Plots a 1xA grid of scalp maps, where A is the number of time windows.
    """
    A_windows = attn_data.shape[1]
    
    # 1 row, A columns. ~5x5 inches per topomap.
    fig, axes = plt.subplots(1, A_windows, figsize=(7.5, 2.5))
    # fig, axes = plt.subplots(1, A_windows, figsize=(7.5, 3.5), gridspec_kw={'wspace': 0.05})
    if A_windows == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Determine Color Scale
    if cmap_type == "absolute":
        # vmin = np.min(attn_data)
        vmin=0
        vmax = np.max(attn_data)
        # cmap = LinearSegmentedColormap.from_list(
        #                                         "peach_to_red",
        #                                         ["#FFCB8D", "#D72638"]
        #                                     )

        cmap = LinearSegmentedColormap.from_list(
            "white_peach_red",
            ["#FEEFDD", "#FFCB8D", "#D72638"]
            )
    else:
        # For difference maps (diverging)
        vmax = np.max(np.abs(attn_data))
        vmin = -vmax
        cmap = plt.get_cmap('bwr')

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for a in range(A_windows):
        ax = axes[a]
        window_weights = attn_data[:, a]
        
        # Draw Head Outline
        head_circle = patches.Circle((0, 0), radius=max_dist, color='black', fill=False, linewidth=1, alpha=0.3)
        ax.add_patch(head_circle)
        
        # Nose (Optional schematic)
        nose_x = [0, max_dist*0.1, -max_dist*0.1, 0]
        nose_y = [max_dist*1.1, max_dist*1.0, max_dist*1.0, max_dist*1.1]
        ax.plot(nose_x, nose_y, color='black', linewidth=1, alpha=0.3)

        # Plot all channels as large colored circles
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=window_weights, cmap=cmap, norm=norm,
                   s=150, edgecolors='white', linewidths=0.1, zorder=2)
        
        # Add text labels inside circles
        for idx in range(len(ch_names)):
            ax.text(coords[idx, 0], coords[idx, 1], ch_names[idx],
                    ha='center', va='center', fontsize=4, color='black' if cmap_type=="bwr" else 'white', 
                    fontweight='bold', zorder=3)
        
        # Formatting
        # ax.set_title(f"Window {a+1}", fontsize=10, pad=10)
        ax.set_title(f"Window {a+1}", fontsize=10, y=-0.15, fontweight='bold', color='black')

        ax.set_aspect('equal')
        ax.axis('off')
        lim_x = max_dist * 1.01
        lim_y = max_dist * 1.15
        ax.set_xlim(-lim_x, lim_x)
        ax.set_ylim(-lim_y, lim_y)

    # # Colorbar layout - adjusted to leave room at the bottom instead of the right
    # fig.tight_layout(rect=[0.0, 0.15, 1.0, 0.9])
    
    # # Add colorbar underneath the plots [left, bottom, width, height]
    # cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.04]) 
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    
    # # Set orientation to horizontal
    # cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    
    # # Remove the black border outline
    # cbar.outline.set_visible(False)
    
    # label = "Attention Weight" if cmap_type == "absolute" else "Δ Attention (Coord - Solo)"
    # cbar.set_label(label, fontsize=12, labelpad=10)

    # 1. Adjust tight_layout to leave room at the TOP [left, bottom, right, top]
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.85])
    
    # 2. Add colorbar near the top [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.1, 0.90, 0.8, 0.04]) 
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Set orientation to horizontal
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    
    # 3. Move ticks and label to the top side of the colorbar
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    
    # Remove the black border outline
    cbar.outline.set_visible(False)
    
    label = "Patch attention weight" if cmap_type == "absolute" else "Δ Attention (Coord - Solo)"
    cbar.set_label(label, fontsize=12, labelpad=8)
    
    # fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_heatmap(data, y_labels, x_labels, title, save_path, cmap="viridis", center=None):
    plt.figure(figsize=(8, 7.5))
    ax = sns.heatmap(data, cmap=cmap, center=center, 
                     yticklabels=y_labels, xticklabels=x_labels, 
                     cbar_kws={'label': 'Attention Weight'})
    
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel("EEG Channels")
    plt.xlabel("Time Windows (1 sec each)")
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_visualizations(data_dir, plot_dir):
    print("--- Loading Saved Data ---")
    
    # Load Channels
    ch_names_path = os.path.join(data_dir, "ch_names.txt")
    with open(ch_names_path, "r") as f:
        ch_names = [line.strip() for line in f.readlines()]
        
    # Load CSVs
    class_0_maps = np.loadtxt(os.path.join(data_dir, "class_0_maps.csv"), delimiter=",")
    class_1_maps = np.loadtxt(os.path.join(data_dir, "class_1_maps.csv"), delimiter=",")

    # If the data happens to be 1D (only 1 window), reshape it so the plotting loops work
    if class_0_maps.ndim == 1:
        class_0_maps = class_0_maps[:, np.newaxis]
        class_1_maps = class_1_maps[:, np.newaxis]

    A = class_0_maps.shape[1]
    time_labels = [f"Win {i+1}" for i in range(A)]
    
    coords, max_dist = get_channel_coords(ch_names)

    print("--- Generating Plots ---")
    os.makedirs(plot_dir, exist_ok=True)

    # Standard Heatmaps
    plot_heatmap(class_0_maps, ch_names, time_labels, "Average Attention: Solo Condition", os.path.join(plot_dir, "attn_matrix_solo.svg"))
    plot_heatmap(class_1_maps, ch_names, time_labels, "Average Attention: Coordination Condition", os.path.join(plot_dir, "attn_matrix_coord.svg"))
    plot_heatmap(class_1_maps - class_0_maps, ch_names, time_labels, "Attention Difference (Coord - Solo)", os.path.join(plot_dir, "attn_matrix_diff.svg"), cmap="coolwarm", center=0)

    # Topomaps
    plot_attention_topomaps(class_0_maps, ch_names, coords, max_dist, "Scalp Attention: Solo", os.path.join(plot_dir, "attn_topo_solo.svg"), cmap_type="absolute")
    plot_attention_topomaps(class_1_maps, ch_names, coords, max_dist, "Scalp Attention: Coordination", os.path.join(plot_dir, "attn_topo_coord.svg"), cmap_type="absolute")
    plot_attention_topomaps(class_1_maps - class_0_maps, ch_names, coords, max_dist, "Scalp Attention Difference (Coord - Solo)", os.path.join(plot_dir, "attn_topo_diff.svg"), cmap_type="diff")

    print(f"All plots saved successfully to {plot_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Extracted Attention Weights")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the extracted CSV/TXT files")
    parser.add_argument("--plot_dir", type=str, default="./attention_plots", help="Directory to save the generated plots")
    args = parser.parse_args()

    generate_visualizations(args.data_dir, args.plot_dir)