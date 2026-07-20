import os
import re
import numpy as np
from glob import glob
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

def extract_segment_number(filepath):
    """Extracts the integer segment number for chronological sorting."""
    # Matches 'seg0.npy', 'seg10.npy', etc.
    match = re.search(r'seg(\d+)\.npy$', filepath)
    return int(match.group(1)) if match else -1

def gather_real_pairs(embeddings_dir):
    """
    Scans the directory and organizes files into a nested dictionary:
    pair_dict[pair_base_name][video_num][participant_id] = [chronological_list_of_files]
    
    Example structure:
    pair_dict['pair10_together_A_10'][3]['p1'] = ['...vid3_seg0.npy', '...vid3_seg1.npy']
    """
    pair_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Recursively find all .npy files
    all_files = glob(os.path.join(embeddings_dir, '**', '*.npy'), recursive=True)
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        
        # Regex to parse the together condition filenames
        # Example: pair10_together_A_10_p1_vid3_seg0.npy
        match = re.match(r'(pair\d+.*)_(p[12])_vid(\d+)_seg\d+\.npy', filename)
        if not match:
            continue
            
        pair_base = match.group(1)   # e.g., 'pair10_together_A_10'
        p_id = match.group(2)        # e.g., 'p1' or 'p2'
        vid_num = int(match.group(3)) # e.g., 3
        
        pair_dict[pair_base][vid_num][p_id].append(filepath)
        
    # Sort the files chronologically by segment number
    for pair_base in pair_dict:
        for vid_num in pair_dict[pair_base]:
            for p_id in ['p1', 'p2']:
                pair_dict[pair_base][vid_num][p_id].sort(key=extract_segment_number)
                
    return pair_dict

def compute_centered_cosine_similarity(timeline_p1, timeline_p2):
    """
    Computes frame-by-frame cosine similarity after mean-centering.
    timeline_p1, timeline_p2: numpy arrays of shape (T, D)
    """
    # 1. Truncate to the shortest timeline to handle potential segment drops
    min_len = min(len(timeline_p1), len(timeline_p2))
    t1 = timeline_p1[:min_len]
    t2 = timeline_p2[:min_len]
    
    # 2. Mean Centering (over the time dimension axis=0)
    # This shifts the origin and solves the transformer anisotropy cone
    t1_centered = t1 - np.mean(t1, axis=0, keepdims=True)
    t2_centered = t2 - np.mean(t2, axis=0, keepdims=True)
    
    # 3. L2 Normalization
    # Add a tiny epsilon to prevent division by zero in flatlines
    eps = 1e-10
    t1_norm = t1_centered / (np.linalg.norm(t1_centered, axis=1, keepdims=True) + eps)
    t2_norm = t2_centered / (np.linalg.norm(t2_centered, axis=1, keepdims=True) + eps)
    
    # 4. Frame-by-frame Cosine Similarity (Dot product along the feature dimension)
    # Resulting shape: (min_len,)
    temporal_isc = np.sum(t1_norm * t2_norm, axis=1)
    
    return temporal_isc

def main():
    # Setup paths
    ENGAGEMENT_DATASET_PATH = os.getenv("ENGAGEMENT_DATASET_PATH", "")
    EMBEDDINGS_DIR = os.path.join(ENGAGEMENT_DATASET_PATH, "embeddings_base")
    OUTPUT_DIR = os.path.join(ENGAGEMENT_DATASET_PATH, "isc_timelines", "real_pairs")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Scanning for real pairs...")
    pair_dict = gather_real_pairs(EMBEDDINGS_DIR)
    
    total_pairs = len(pair_dict)
    print(f"Found {total_pairs} valid real pairs.")
    
    for pair_base, videos in pair_dict.items():
        for vid_num, participants in videos.items():
            
            p1_files = participants.get('p1', [])
            p2_files = participants.get('p2', [])
            
            # Skip if one partner is entirely missing for this video
            if not p1_files or not p2_files:
                print(f"Warning: Missing partner for {pair_base} video {vid_num}. Skipping.")
                continue
            
            # Load timelines into memory (T, D)
            try:
                timeline_p1 = np.array([np.load(f) for f in p1_files])
                timeline_p2 = np.array([np.load(f) for f in p2_files])
            except Exception as e:
                print(f"Error loading arrays for {pair_base} vid {vid_num}: {e}")
                continue
                
            # Compute ISC
            isc_timeline = compute_centered_cosine_similarity(timeline_p1, timeline_p2)
            
            # Save the timeline
            # Filename example: isc_pair10_together_A_10_vid3.npy
            save_name = f"isc_{pair_base}_vid{vid_num}.npy"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            np.save(save_path, isc_timeline)
            
        print(f"Processed ISC for pair: {pair_base}")

    print(f"\nSuccess! All real pair ISC timelines saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()