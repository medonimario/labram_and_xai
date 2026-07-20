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

def compute_subject_global_mean(pair_base, p_id, pair_dict):
    """
    Pass 1: Computes the global mean embedding vector for a single participant 
    across all videos they watched.
    Returns: A 1D numpy array of shape (D,) representing the subject's true center.
    """
    all_embeddings = []
    
    # Iterate through all videos for this specific pair
    for vid_num, participants in pair_dict[pair_base].items():
        files = participants.get(p_id, [])
        for f in files:
            all_embeddings.append(np.load(f))
            
    if not all_embeddings:
        return None
        
    # Stack into shape (Total_Segments, D) and compute mean over axis 0
    stacked = np.vstack(all_embeddings)
    subject_mean = np.mean(stacked, axis=0)
    return subject_mean

def compute_isc(timeline_p1, timeline_p2, mu_p1, mu_p2):
    """
    Pass 2: Computes frame-by-frame cosine similarity using the pre-computed 
    global subject means to shift the origin.
    """
    min_len = min(len(timeline_p1), len(timeline_p2))
    t1 = timeline_p1[:min_len]
    t2 = timeline_p2[:min_len]
    
    # Center using the Subject's Global Mean (Preserves cross-video absolute space)
    t1_centered = t1 - mu_p1
    t2_centered = t2 - mu_p2
    
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
    OUTPUT_DIR = os.path.join(ENGAGEMENT_DATASET_PATH, "isc_timelines", "real_pairs_subject_centered")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Scanning for real pairs...")
    pair_dict = gather_real_pairs(EMBEDDINGS_DIR)
    print(f"Found {len(pair_dict)} valid real pairs.")
    
    for pair_base, videos in pair_dict.items():
        print(f"Processing {pair_base}...")
        
        # --- PASS 1: Calculate Global Subject Means ---
        mu_p1 = compute_subject_global_mean(pair_base, 'p1', pair_dict)
        mu_p2 = compute_subject_global_mean(pair_base, 'p2', pair_dict)
        
        if mu_p1 is None or mu_p2 is None:
            print(f"  Warning: Incomplete data for {pair_base} to compute global mean. Skipping.")
            continue
            
        # --- PASS 2: Compute ISC per video ---
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
                
            isc_timeline = compute_isc(timeline_p1, timeline_p2, mu_p1, mu_p2)
            
            save_name = f"isc_{pair_base}_vid{vid_num}.npy"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            np.save(save_path, isc_timeline)

    print(f"\nSuccess! All properly centered ISC timelines saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()