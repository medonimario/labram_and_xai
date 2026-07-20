import os
import re
import numpy as np
from glob import glob
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

def extract_segment_number(filepath):
    """Extracts the integer segment number for chronological sorting."""
    match = re.search(r'seg(\d+)\.npy$', filepath)
    return int(match.group(1)) if match else -1

def gather_real_pairs(embeddings_dir):
    """
    Organizes files into pair_dict[pair_base_name][video_num][participant_id] = [files]
    """
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

def compute_global_mean(embeddings_dir):
    """
    Pass 1: Computes the grand global mean of ALL embeddings in the dataset.
    Uses a running sum to be highly memory efficient.
    """
    print("Computing the Global Mean across all subjects and videos...")
    all_files = glob(os.path.join(embeddings_dir, '**', '*.npy'), recursive=True)
    
    if not all_files:
        raise ValueError(f"No .npy files found in {embeddings_dir}")
        
    # Initialize variables to accumulate the sum
    total_sum = None
    total_count = 0
    
    for filepath in all_files:
        emb = np.load(filepath)
        if total_sum is None:
            total_sum = np.zeros_like(emb, dtype=np.float64)
            
        total_sum += emb
        total_count += 1
        
    mu_global = total_sum / total_count
    print(f"  -> Processed {total_count} total segments.")
    print(f"  -> Global Mean vector shape: {mu_global.shape}")
    
    return mu_global

def compute_isc_global(timeline_p1, timeline_p2, mu_global):
    """
    Pass 2: Computes frame-by-frame cosine similarity using the grand 
    global mean to shift the origin.
    """
    min_len = min(len(timeline_p1), len(timeline_p2))
    t1 = timeline_p1[:min_len]
    t2 = timeline_p2[:min_len]
    
    # Center using the Grand Global Mean
    t1_centered = t1 - mu_global
    t2_centered = t2 - mu_global
    
    # L2 Normalization
    eps = 1e-10
    t1_norm = t1_centered / (np.linalg.norm(t1_centered, axis=1, keepdims=True) + eps)
    t2_norm = t2_centered / (np.linalg.norm(t2_centered, axis=1, keepdims=True) + eps)
    
    # Frame-by-frame Cosine Similarity
    temporal_isc = np.sum(t1_norm * t2_norm, axis=1)
    return temporal_isc

def main():
    ENGAGEMENT_DATASET_PATH = os.getenv("ENGAGEMENT_DATASET_PATH", "")
    EMBEDDINGS_DIR = os.path.join(ENGAGEMENT_DATASET_PATH, "embeddings_base")
    
    # Save to a distinct folder so you can compare plotting later
    OUTPUT_DIR = os.path.join(ENGAGEMENT_DATASET_PATH, "isc_timelines", "real_pairs_global_centered")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- PASS 1: Calculate Grand Global Mean ---
    mu_global = compute_global_mean(EMBEDDINGS_DIR)
    
    print("\nScanning for real pairs...")
    pair_dict = gather_real_pairs(EMBEDDINGS_DIR)
    print(f"Found {len(pair_dict)} valid real pairs.")
    
    # --- PASS 2: Compute ISC per video ---
    for pair_base, videos in pair_dict.items():
        print(f"Processing {pair_base}...")
        
        for vid_num, participants in videos.items():
            p1_files = participants.get('p1', [])
            p2_files = participants.get('p2', [])
            
            if not p1_files or not p2_files:
                continue
                
            try:
                timeline_p1 = np.array([np.load(f) for f in p1_files])
                timeline_p2 = np.array([np.load(f) for f in p2_files])
            except Exception as e:
                print(f"  Error loading arrays for vid {vid_num}: {e}")
                continue
                
            # Use the global mean here
            isc_timeline = compute_isc_global(timeline_p1, timeline_p2, mu_global)
            
            save_name = f"isc_{pair_base}_vid{vid_num}.npy"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            np.save(save_path, isc_timeline)

    print(f"\nSuccess! All globally-centered ISC timelines saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()