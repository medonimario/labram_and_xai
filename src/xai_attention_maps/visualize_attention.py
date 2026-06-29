import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from torch.nn import functional as F
from einops import rearrange
import mne
from mne.viz.topomap import _find_topomap_coords

# Import your existing utilities
# from src.labram_ft.run_class_finetuning import get_dataset, get_models
# import src.labram_ft.utils
# Compute paths relative to this script's location
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))          # Adds 'src/'
sys.path.append(os.path.abspath(os.path.join(current_dir, "../labram_ft"))) # Adds 'src/labram_ft/'

# Now your imports work perfectly without needing python -m or environment variables
from labram_ft.run_class_finetuning import get_dataset, get_models
import utils

def get_channel_coords(ch_names):
    """
    Returns 2D coordinates for the dataset channel list using MNE.
    Automatically handles capitalization differences (e.g., 'FPZ' -> 'Fpz')
    """
    # Fix casing for MNE standard 10-20/10-05 montages
    mne_names = [ch.replace('Z', 'z').replace('FP', 'Fp') for ch in ch_names]
    
    montage = mne.channels.make_standard_montage('standard_1020')
    
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
    attn_data should be shape [N_channels, A_windows]
    """
    A_windows = attn_data.shape[1]
    
    # 1 row, A columns. ~5x5 inches per topomap.
    fig, axes = plt.subplots(1, A_windows, figsize=(5 * A_windows, 5))
    if A_windows == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Determine Color Scale
    if cmap_type == "absolute":
        vmin = 0
        vmax = np.max(attn_data)
        cmap = plt.get_cmap('plasma')
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
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=window_weights, cmap=cmap, norm=norm,
                        s=450, edgecolors='grey', linewidths=0.1, zorder=2)
        
        # Add text labels inside circles
        for idx in range(len(ch_names)):
            ax.text(coords[idx, 0], coords[idx, 1], ch_names[idx],
                    ha='center', va='center', fontsize=8, color='black' if cmap_type=="absolute" else 'white', 
                    fontweight='bold', zorder=3)
        
        # Formatting
        ax.set_title(f"Time Window {a+1} (sec)", fontsize=14, pad=10)
        ax.set_aspect('equal')
        ax.axis('off')
        lim = max_dist * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    # Colorbar layout
    fig.tight_layout(rect=[0.0, 0.0, 0.92, 0.9])
    
    # Add colorbar on the right
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    label = "Attention Weight" if cmap_type == "absolute" else "Δ Attention (Coord - Solo)"
    cbar.set_label(label, fontsize=14)
    
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_heatmap(data, y_labels, x_labels, title, save_path, cmap="viridis", center=None):
    plt.figure(figsize=(8, 14))
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


def extract_and_plot_attention(checkpoint_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint-best.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")

    print("--- Loading Checkpoint and Arguments ---")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args = checkpoint['args']
    args.use_attention_pooling = True 
    print(f"Dataset: {args.dataset}")
    
    # 1. Load Dataset
    _, test_dataset, _, ch_names, _ = get_dataset(args)
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 2. Get MNE Coordinates for Topomaps
    coords, max_dist = get_channel_coords(ch_names)

    # 3. Load Model
    model = get_models(args)
    utils.load_state_dict(model, checkpoint.get('model', checkpoint.get('module', checkpoint)))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 4. Setup Forward Hook
    saved_attn_logits = []
    def hook_fn(module, input, output):
        saved_attn_logits.append(output.detach())
    handle = model.attention_pool.register_forward_hook(hook_fn)
    input_chans = utils.get_input_chans(ch_names)

    all_targets = []

    print("--- Extracting Attention Weights ---")
    with torch.no_grad():
        for batch in data_loader_test:
            EEG = batch[0].float().to(device) / 100
            EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200) 
            target = batch[-1].to(device)
            
            _ = model(EEG, input_chans=input_chans)
            print(f"Logits collected so far: {len(saved_attn_logits)}")
            all_targets.append(target.cpu().numpy())

    handle.remove()

    # 5. Process Logits into Softmax
    all_attn_logits = torch.cat(saved_attn_logits, dim=0)          
    all_attn_weights = F.softmax(all_attn_logits, dim=1).cpu().numpy()
    all_targets = np.concatenate(all_targets, axis=0)              

    B, N_A, _ = all_attn_weights.shape
    N = len(ch_names)       
    A = N_A // N            
    
    attn_maps = all_attn_weights.reshape(B, N, A)

    print("--- Plotting Results ---")
    plot_dir = os.path.join(checkpoint_dir, "attention_plots")
    os.makedirs(plot_dir, exist_ok=True)

    if all_targets.ndim > 1:
        all_targets = all_targets.squeeze()
        
    class_0_maps = attn_maps[all_targets == 0].mean(axis=0) 
    class_1_maps = attn_maps[all_targets == 1].mean(axis=0) 
    
    time_labels = [f"Win {i+1}" for i in range(A)]
    
    # Standard Heatmaps
    plot_heatmap(class_0_maps, ch_names, time_labels, "Average Attention: Solo Condition", os.path.join(plot_dir, "attn_matrix_solo.png"))
    plot_heatmap(class_1_maps, ch_names, time_labels, "Average Attention: Coordination Condition", os.path.join(plot_dir, "attn_matrix_coord.png"))
    plot_heatmap(class_1_maps - class_0_maps, ch_names, time_labels, "Attention Difference (Coord - Solo)", os.path.join(plot_dir, "attn_matrix_diff.png"), cmap="coolwarm", center=0)

    # Topomaps (Using your style)
    plot_attention_topomaps(class_0_maps, ch_names, coords, max_dist, "Scalp Attention: Solo", os.path.join(plot_dir, "attn_topo_solo.png"), cmap_type="absolute")
    plot_attention_topomaps(class_1_maps, ch_names, coords, max_dist, "Scalp Attention: Coordination", os.path.join(plot_dir, "attn_topo_coord.png"), cmap_type="absolute")
    plot_attention_topomaps(class_1_maps - class_0_maps, ch_names, coords, max_dist, "Scalp Attention Difference (Coord - Solo)", os.path.join(plot_dir, "attn_topo_diff.png"), cmap_type="diff")

    print(f"All plots saved to {plot_dir}")

if __name__ == "__main__":
    TARGET_DIR = "./src/labram_ft/checkpoints/fg_v1/"
    extract_and_plot_attention(TARGET_DIR)