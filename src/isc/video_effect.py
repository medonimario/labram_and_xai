import os
import re
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Parameters
N_PERMUTATIONS = 1000
MIN_LAG_SECONDS = 30.0   # Minimum shift to ensure we break time-locking
STEP_SIZE_SEC = 2.0      # 2s step in your windows
MIN_LAG_FRAMES = int(MIN_LAG_SECONDS / STEP_SIZE_SEC) 

def extract_segment_number(filepath):
    match = re.search(r'seg(\d+)\.npy$', filepath)
    return int(match.group(1)) if match else -1

def gather_real_pairs(embeddings_dir):
    """Same gathering logic as before."""
    pair_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_files = glob(os.path.join(embeddings_dir, '**', '*.npy'), recursive=True)
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        match = re.match(r'(pair\d+.*)_(p[12])_vid(\d+)_seg\d+\.npy', filename)
        if not match: continue
        pair_dict[match.group(1)][int(match.group(3))][match.group(2)].append(filepath)
        
    for pb in pair_dict:
        for vid in pair_dict[pb]:
            for p_id in ['p1', 'p2']:
                pair_dict[pb][vid][p_id].sort(key=extract_segment_number)
    return pair_dict

def compute_mean_isc(t1, t2):
    """Computes the scalar mean ISC for two timelines."""
    # Mean Centering
    t1_c = t1 - np.mean(t1, axis=0, keepdims=True)
    t2_c = t2 - np.mean(t2, axis=0, keepdims=True)
    
    # L2 Norm
    eps = 1e-10
    t1_n = t1_c / (np.linalg.norm(t1_c, axis=1, keepdims=True) + eps)
    t2_n = t2_c / (np.linalg.norm(t2_c, axis=1, keepdims=True) + eps)
    
    # Dot product & mean
    return np.mean(np.sum(t1_n * t2_n, axis=1))

def main():
    ENGAGEMENT_DATASET_PATH = os.getenv("ENGAGEMENT_DATASET_PATH", "")
    EMBEDDINGS_DIR = os.path.join(ENGAGEMENT_DATASET_PATH, "embeddings_base")
    OUTPUT_CSV = "./src/isc/isc_statistics_within_video.csv"
    
    print("Gathering files...")
    pair_dict = gather_real_pairs(EMBEDDINGS_DIR)
    
    results = []
    
    print(f"Running Circular Shift Permutations ({N_PERMUTATIONS} iterations per pair/video)...")
    
    for pair_base, videos in tqdm(pair_dict.items(), desc="Pairs"):
        for vid_num, participants in videos.items():
            p1_files = participants.get('p1', [])
            p2_files = participants.get('p2', [])
            
            if not p1_files or not p2_files:
                continue
                
            try:
                timeline_p1 = np.array([np.load(f) for f in p1_files])
                timeline_p2 = np.array([np.load(f) for f in p2_files])
            except Exception:
                continue
            
            # Truncate to min length
            T = min(len(timeline_p1), len(timeline_p2))
            t1 = timeline_p1[:T]
            t2 = timeline_p2[:T]
            
            # If the video is too short to allow for our minimum lag, skip it
            if T <= MIN_LAG_FRAMES * 2:
                print(f"Skipping {pair_base} Vid {vid_num}: Timeline too short for valid lags.")
                continue

            # 1. Compute True ISC
            true_isc = compute_mean_isc(t1, t2)
            
            # 2. Build Null Distribution via Circular Shifting
            null_distribution = np.zeros(N_PERMUTATIONS)
            
            # Generate random lags, excluding the zone too close to 0 (or T)
            valid_lags = np.arange(MIN_LAG_FRAMES, T - MIN_LAG_FRAMES)
            
            # If for some reason valid_lags is empty due to length
            if len(valid_lags) == 0:
                 continue
                 
            random_lags = np.random.choice(valid_lags, size=N_PERMUTATIONS, replace=True)
            
            for i, lag in enumerate(random_lags):
                # Circularly shift Subject 2
                t2_shifted = np.roll(t2, shift=lag, axis=0)
                null_distribution[i] = compute_mean_isc(t1, t2_shifted)
                
            # 3. Calculate p-value
            # Formula: (Number of nulls >= true_isc + 1) / (Total Perms + 1)
            # The +1 is standard practice to prevent p=0 exactly.
            count_greater = np.sum(null_distribution >= true_isc)
            p_value = (count_greater + 1) / (N_PERMUTATIONS + 1)
            
            results.append({
                'Pair': pair_base,
                'Video': vid_num,
                'True_ISC': true_isc,
                'Null_Mean': np.mean(null_distribution),
                'Null_Std': np.std(null_distribution),
                'p_value': p_value,
                'Significant_05': p_value <= 0.05
            })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCompleted! Statistics saved to: {OUTPUT_CSV}")
    
    # Print a quick summary
    sig_count = df['Significant_05'].sum()
    total = len(df)
    print("\n--- SUMMARY ---")
    print(f"Total Video/Pair Combinations Evaluated: {total}")
    print(f"Number of Significantly Synchronized (p <= 0.05): {sig_count} ({(sig_count/total)*100:.1f}%)")

if __name__ == "__main__":
    main()