import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from dotenv import load_dotenv

load_dotenv()

def extract_segment_number(filepath):
    match = re.search(r'seg(\d+)\.npy$', filepath)
    return int(match.group(1)) if match else -1

def gather_real_pairs(embeddings_dir):
    pair_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_files = glob(os.path.join(embeddings_dir, '**', '*.npy'), recursive=True)
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        match = re.match(r'(pair\d+.*)_(p[12])_vid(\d+)_seg\d+\.npy', filename)
        if not match:
            continue
            
        pair_base = match.group(1)
        p_id = match.group(2)
        vid_num = int(match.group(3))
        
        pair_dict[pair_base][vid_num][p_id].append(filepath)
        
    for pair_base in pair_dict:
        for vid_num in pair_dict[pair_base]:
            for p_id in ['p1', 'p2']:
                pair_dict[pair_base][vid_num][p_id].sort(key=extract_segment_number)
                
    return pair_dict

def main():
    ENGAGEMENT_DATASET_PATH = os.getenv("ENGAGEMENT_DATASET_PATH", "")
    EMBEDDINGS_DIR = os.path.join(ENGAGEMENT_DATASET_PATH, "embeddings_base")
    
    OUTPUT_PLOTS_DIR = "./src/isc/isc_rsa_plots"
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    
    print("Scanning for real pairs...")
    pair_dict = gather_real_pairs(EMBEDDINGS_DIR)
    
    results = []
    
    for pair_base, videos in pair_dict.items():
        for vid_num, participants in videos.items():
            
            p1_files = participants.get('p1', [])
            p2_files = participants.get('p2', [])
            
            if not p1_files or not p2_files:
                continue
                
            try:
                timeline_p1 = np.array([np.load(f) for f in p1_files])
                timeline_p2 = np.array([np.load(f) for f in p2_files])
            except Exception as e:
                print(f"Error loading arrays for {pair_base} vid {vid_num}: {e}")
                continue
                
            # 1. Truncate to matching length
            min_len = min(len(timeline_p1), len(timeline_p2))
            t1 = timeline_p1[:min_len]
            t2 = timeline_p2[:min_len]
            
            # 2. Compute the condensed Time-by-Time distance matrices (RDMs)
            # Using 'correlation' distance (1 - Pearson r) as is standard in RSA to center features
            rdm_t1_condensed = pdist(t1, metric='correlation')
            rdm_t2_condensed = pdist(t2, metric='correlation')
            
            # 3. Compute IS-RSA (Spearman rank correlation of the upper triangles)
            # We add a tiny jitter to avoid Spearman warnings if distances are perfectly tied
            rho, p_val = spearmanr(rdm_t1_condensed, rdm_t2_condensed)
            
            results.append({
                'Pair': pair_base,
                'Video': vid_num,
                'IS_RSA_rho': rho,
                'p_value': p_val,
                'Segments': min_len
            })
            
            # --- 4. Plotting the matrices for visual inspection ---
            # Convert condensed 1D arrays back to 2D T x T matrices for plotting
            rdm_t1_sq = squareform(rdm_t1_condensed)
            rdm_t2_sq = squareform(rdm_t2_condensed)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # We plot 1 - distance to show similarity (brighter = more similar state)
            sns.heatmap(1 - rdm_t1_sq, ax=axes[0], cmap='magma', square=True, 
                        cbar_kws={'label': 'Pearson Correlation'})
            axes[0].set_title("Participant 1 (P1) RDM")
            axes[0].set_xlabel("Time Window Index")
            axes[0].set_ylabel("Time Window Index")
            
            sns.heatmap(1 - rdm_t2_sq, ax=axes[1], cmap='magma', square=True, 
                        cbar_kws={'label': 'Pearson Correlation'})
            axes[1].set_title("Participant 2 (P2) RDM")
            axes[1].set_xlabel("Time Window Index")
            
            fig.suptitle(f"{pair_base} | Video {vid_num}\nIS-RSA Alignment: $\\rho$ = {rho:.4f} (p={p_val:.2e})", 
                         fontsize=14, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_PLOTS_DIR, f"rsa_{pair_base}_vid{vid_num}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Processed IS-RSA for {pair_base} Video {vid_num} -> rho: {rho:.3f}")

    # 5. Save Summary DataFrame
    df = pd.DataFrame(results)
    csv_path = os.path.join(ENGAGEMENT_DATASET_PATH, "is_rsa_results_summary.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n======================================")
    print(f"IS-RSA computation complete!")
    print(f"-> All heatmaps saved to: {OUTPUT_PLOTS_DIR}")
    print(f"-> Summary CSV saved to: {csv_path}")
    print(f"Average IS-RSA across all pairs/videos: {df['IS_RSA_rho'].mean():.4f}")

if __name__ == "__main__":
    main()