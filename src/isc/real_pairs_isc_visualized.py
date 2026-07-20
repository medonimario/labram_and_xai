import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# The noise mapping based on the dataset README
NOISE_MAPPING = {
    0:  {'A': 'No Noise',   'B': 'No Noise'},
    3:  {'A': 'Low Noise',  'B': 'High Noise'},
    4:  {'A': 'Low Noise',  'B': 'High Noise'},
    5:  {'A': 'Low Noise',  'B': 'High Noise'},
    6:  {'A': 'Low Noise',  'B': 'High Noise'},
    7:  {'A': 'High Noise', 'B': 'Low Noise'},
    8:  {'A': 'High Noise', 'B': 'Low Noise'},
    9:  {'A': 'High Noise', 'B': 'Low Noise'},
    10: {'A': 'High Noise', 'B': 'Low Noise'}
}

# Windowing parameters
STEP_SIZE_SEC = 2.0  # 4s window with 2s overlap = 2s step

def main():
    ENGAGEMENT_DATASET_PATH = os.getenv("ENGAGEMENT_DATASET_PATH", "")
    INPUT_DIR = os.path.join(ENGAGEMENT_DATASET_PATH, "isc_timelines", "real_pairs_global_centered")
    OUTPUT_DIR = "./src/isc/plots_global_centered"  # Save plots in the local src/isc directory
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Structure: data[video_id][pair_base_name] = {'timeline': array, 'condition': A/B, 'noise': Low/High}
    video_data = defaultdict(dict)
    
    # 1. Parse files and organize data
    all_files = glob(os.path.join(INPUT_DIR, "*.npy"))
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        # Expected format: isc_pair10_together_A_10_vid3.npy
        match = re.match(r'isc_(pair\d+_together_([AB])_\d+)_vid(\d+)\.npy', filename)
        if not match:
            continue
            
        pair_base = match.group(1)
        condition = match.group(2)
        vid_num = int(match.group(3))
        
        timeline = np.load(filepath)
        noise_level = NOISE_MAPPING.get(vid_num, {}).get(condition, "Unknown")
        
        video_data[vid_num][pair_base] = {
            'timeline': timeline,
            'condition': condition,
            'noise': noise_level
        }

    # 2. Plotting and Printing Stats
    print("=" * 60)
    print("AVERAGE ISC VALUES PER PAIR AND VIDEO")
    print("=" * 60)
    
    for vid_num in sorted(video_data.keys()):
        pairs = video_data[vid_num]
        print(f"\n--- VIDEO {vid_num} ---")
        
        # Prepare data for averages
        low_noise_timelines = []
        high_noise_timelines = []
        no_noise_timelines = [] # Specifically for Video 0
        
        # Calculate grid size: N pairs + 2 averages
        num_subplots = len(pairs) + 2
        cols = 4
        rows = math.ceil(num_subplots / cols)
        
        # Create figure (sharex=False as requested)
        fig, axes = plt.subplots(rows, cols, figsize=(24, 4 * rows), sharex=False)
        fig.suptitle(f"ISC Timelines - Video {vid_num}", fontsize=20, fontweight='bold', y=1.02)
        axes = axes.flatten()
        
        # Track subplot index
        ax_idx = 0
        
        for pair_base, data in pairs.items():
            timeline = data['timeline']
            noise_level = data['noise']
            
            # Print console stats
            mean_isc = np.mean(timeline)
            print(f"{pair_base} ({noise_level}): {mean_isc:.4f}")
            
            # Group for averages
            if noise_level == 'Low Noise':
                low_noise_timelines.append(timeline)
            elif noise_level == 'High Noise':
                high_noise_timelines.append(timeline)
            elif noise_level == 'No Noise':
                no_noise_timelines.append(timeline)
                
            # Plot individual pair
            ax = axes[ax_idx]
            time_sec = np.arange(len(timeline)) * STEP_SIZE_SEC
            
            # Color code based on noise
            color = 'tab:blue' if noise_level == 'Low Noise' else ('tab:red' if noise_level == 'High Noise' else 'tab:green')
            
            ax.plot(time_sec, timeline, color=color, linewidth=1.5)
            ax.set_title(f"{pair_base}\n({noise_level})", fontsize=10)
            ax.set_xlabel("Window Start Time (s)")
            ax.set_ylabel("Cosine Similarity")
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Dynamically set y-limits based on data to see fluctuations better
            ax.set_ylim([min(-1.0, np.min(timeline) - 0.1), max(1.0, np.max(timeline) + 0.1)]) 
            
            ax_idx += 1
            
        # 3. Plot Averages
        def pad_and_average(timelines):
            """Helper to average arrays of potentially slightly different lengths."""
            if not timelines:
                return np.array([])
            max_len = max(len(t) for t in timelines)
            padded = np.array([np.pad(t, (0, max_len - len(t)), constant_values=np.nan) for t in timelines])
            return np.nanmean(padded, axis=0)

        # Average Plot 1 (Low Noise / No Noise A)
        ax_avg1 = axes[ax_idx]
        if vid_num == 0:
            avg_no_noise = pad_and_average(no_noise_timelines)
            time_sec = np.arange(len(avg_no_noise)) * STEP_SIZE_SEC
            ax_avg1.plot(time_sec, avg_no_noise, color='tab:green', linewidth=2.5)
            ax_avg1.set_title("AVERAGE: No Noise", fontsize=12, fontweight='bold')
        else:
            avg_low = pad_and_average(low_noise_timelines)
            time_sec = np.arange(len(avg_low)) * STEP_SIZE_SEC
            ax_avg1.plot(time_sec, avg_low, color='tab:blue', linewidth=2.5)
            ax_avg1.set_title("AVERAGE: Low Noise", fontsize=12, fontweight='bold')
            
        ax_avg1.set_xlabel("Window Start Time (s)")
        ax_avg1.set_ylabel("Mean Cosine Similarity")
        ax_avg1.grid(True, linestyle='--', alpha=0.6)
        ax_idx += 1

        # Average Plot 2 (High Noise)
        ax_avg2 = axes[ax_idx]
        if vid_num != 0:
            avg_high = pad_and_average(high_noise_timelines)
            time_sec = np.arange(len(avg_high)) * STEP_SIZE_SEC
            ax_avg2.plot(time_sec, avg_high, color='tab:red', linewidth=2.5)
            ax_avg2.set_title("AVERAGE: High Noise", fontsize=12, fontweight='bold')
            ax_avg2.set_xlabel("Window Start Time (s)")
            ax_avg2.set_ylabel("Mean Cosine Similarity")
            ax_avg2.grid(True, linestyle='--', alpha=0.6)
        else:
            # Hide the extra axis for Video 0 if not needed, or plot something else
            ax_avg2.axis('off')
            
        ax_idx += 1
        
        # Hide any remaining unused subplots
        for i in range(ax_idx, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(OUTPUT_DIR, f"isc_grid_vid{vid_num}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"-> Saved plot for Video {vid_num} to {save_path}")

    print("\nAll plotting complete!")

if __name__ == "__main__":
    main()