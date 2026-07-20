import os
import re
import itertools
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from dotenv import load_dotenv

load_dotenv()

def extract_segment_number(filepath):
    match = re.search(r'seg(\d+)\.npy$', filepath)
    return int(match.group(1)) if match else -1

def gather_solo_subjects(embeddings_dir):
    """
    Scans the directory and organizes solo files:
    solo_dict[video_num][solo_id] = [chronological_list_of_files]
    """
    solo_dict = defaultdict(lambda: defaultdict(list))
    all_files = glob(os.path.join(embeddings_dir, '**', '*.npy'), recursive=True)
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        # Match solo condition filenames
        # Example: solo10_A_10_vid3_seg0.npy
        match = re.match(r'(solo\d+_[AB]_\d+)_vid(\d+)_seg\d+\.npy', filename)
        if not match:
            continue
            
        solo_id = match.group(1)
        vid_num = int(match.group(2))
        
        solo_dict[vid_num][solo_id].append(filepath)
        
    # Sort chronologically
    for vid_num in solo_dict:
        for solo_id in solo_dict[vid_num]:
            solo_dict[vid_num][solo_id].sort(key=extract_segment_number)
                
    return solo_dict

def main():
    ENGAGEMENT_DATASET_PATH = os.getenv("ENGAGEMENT_DATASET_PATH", "")
    EMBEDDINGS_DIR = os.path.join(ENGAGEMENT_DATASET_PATH, "embeddings_base")
    
    print("Scanning for solo subjects to form pseudo-pairs...")
    solo_dict = gather_solo_subjects(EMBEDDINGS_DIR)
    
    results = []
    total_pseudo_pairs = 0
    
    # Process video by video
    for vid_num, subjects in solo_dict.items():
        subject_ids = list(subjects.keys())
        
        # Create all unique combinations of solo subjects for this video
        # E.g., if there are 40 solo subjects, this creates (40 * 39) / 2 = 780 pairs per video
        pseudo_pairs = list(itertools.combinations(subject_ids, 2))
        print(f"Video {vid_num}: Computing IS-RSA for {len(pseudo_pairs)} pseudo-pairs...")
        
        for p1_id, p2_id in pseudo_pairs:
            p1_files = subjects[p1_id]
            p2_files = subjects[p2_id]
            
            if not p1_files or not p2_files:
                continue
                
            try:
                timeline_p1 = np.array([np.load(f) for f in p1_files])
                timeline_p2 = np.array([np.load(f) for f in p2_files])
            except Exception as e:
                continue
                
            # 1. Truncate to matching length
            min_len = min(len(timeline_p1), len(timeline_p2))
            t1 = timeline_p1[:min_len]
            t2 = timeline_p2[:min_len]
            
            # Skip if sequence is too short to correlate meaningfully
            if min_len < 5: 
                continue
            
            # 2. Compute RDMs
            rdm_t1_condensed = pdist(t1, metric='correlation')
            rdm_t2_condensed = pdist(t2, metric='correlation')
            
            # 3. Compute IS-RSA
            rho, p_val = spearmanr(rdm_t1_condensed, rdm_t2_condensed)
            
            results.append({
                'Pseudo_Pair': f"{p1_id}_AND_{p2_id}",
                'Video': vid_num,
                'IS_RSA_rho': rho,
                'p_value': p_val,
                'Segments': min_len
            })
            total_pseudo_pairs += 1

    # Save Summary DataFrame
    df = pd.DataFrame(results)
    csv_path = os.path.join(ENGAGEMENT_DATASET_PATH, "is_rsa_pseudo_results_summary.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n======================================")
    print(f"IS-RSA Pseudo-Pair computation complete!")
    print(f"-> Summary CSV saved to: {csv_path}")
    print(f"Total Pseudo-Pairs processed: {total_pseudo_pairs}")
    
    # Handle NaNs gracefully if any perfect ties resulted in NaN spearman
    valid_rhos = df['IS_RSA_rho'].dropna()
    print(f"Average Pseudo-Pair IS-RSA across all videos: {valid_rhos.mean():.4f}")

if __name__ == "__main__":
    main()