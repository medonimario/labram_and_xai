import os
import re
import pickle
import numpy as np
import mne
from multiprocessing import Pool
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# Standard channel order expected by the model
standard_channels = [
    'Fp1','Fpz','Fp2',
    'AF7','AF3','AFz','AF4','AF8',
    'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8',
    'P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10',
    'PO7','PO3','POz','PO4','PO8',
    'O1','Oz','O2',
    'Iz'
]

# Faulty subjects to exclude
BLOCKLIST = ['solo11_A_1']

def get_label(entity_id):
    """
    Assigns binary label based on the entity type (Condition).
    Label 0: Solo
    Label 1: Together (Pair)
    """
    if 'solo' in entity_id:
        return 0
    elif 'pair' in entity_id:
        return 1
    else:
        raise ValueError(f"Unknown condition for entity: {entity_id}")

def process_npz_file(params):
    """
    Worker function to process a single .npz file.
    Resamples, crops into 4s overlapping windows, and saves.
    """
    filepath, dump_folder, split_name, entity_id = params
    
    try:
        # 1. Load data
        f = np.load(filepath, allow_pickle=True)
        video_num = int(f["video_number"])
        sfreq_orig = float(f["sfreq"])
        ch_names = [str(c) for c in f["ch_names"]]
        
        # Transpose to (n_channels, n_samples)
        data_volts = f["data"].T 
        
        # 2. Get Label
        label = get_label(entity_id)
        
        # 3. MNE Processing (Resampling & Unit Conversion)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq_orig, ch_types='eeg')
        raw = mne.io.RawArray(data_volts, info, verbose=False)
        
        if sfreq_orig != 200.0:
            raw.resample(200.0, n_jobs=1, verbose=False)
            
        # Reorder channels safely
        if all(ch in raw.ch_names for ch in standard_channels):
            raw.reorder_channels(standard_channels)
        else:
            print(f"Warning: Channel mismatch in {filepath}. Skipping.")
            return None

        # Extract data in microvolts (uV)
        data_uV = raw.get_data(units='uV')
        
        # 4. Segmenting using Sliding Window
        sfreq_new = 200.0
        segm_len_samples = int(4.0 * sfreq_new)  # 800 samples
        overlap_samples = int(2.0 * sfreq_new)   # 400 samples
        step_size_samples = segm_len_samples - overlap_samples
        n_samples = data_uV.shape[1]
        
        base_filename = os.path.basename(os.path.dirname(filepath)) # e.g. solo10_A_10
        n_segments = 0
        
        for start_idx in range(0, n_samples - segm_len_samples + 1, step_size_samples):
            end_idx = start_idx + segm_len_samples
            segment_data = data_uV[:, start_idx:end_idx]
            
            dump_path = os.path.join(
                dump_folder, 
                f"{base_filename}_vid{video_num}_seg{n_segments}.pkl"
            )
            
            pickle.dump(
                {"X": segment_data, "y": label},
                open(dump_path, "wb")
            )
            n_segments += 1
            
        return (split_name, entity_id, video_num, n_segments)

    except Exception as e:
        print(f"ERROR processing file {filepath}: {e}")
        with open("engagement_dataset_errors.txt", "a") as err_f:
            err_f.write(f"{filepath} - {e}\n")
        return None

def main():
    # Set paths based on your environment
    ENGAGEMENT_DATASET_PATH = os.getenv("ENGAGEMENT_DATASET_PATH", "")
    base_path = os.path.join(ENGAGEMENT_DATASET_PATH, "data_for_mario")
    final_root = os.path.join(ENGAGEMENT_DATASET_PATH, "processed")
    
    # Dictionaries to group files
    entity_files = defaultdict(list)
    
    print("Scanning directories for .npz files...")
    for root, _, files in os.walk(base_path):
        folder_name = os.path.basename(root)
        
        # Check blocklist
        if any(bad_subj in folder_name for bad_subj in BLOCKLIST):
            continue
            
        for file in files:
            if file.endswith('.npz'):
                # Ignore video0 (pilot, no noise)
                if file == 'video0.npz':
                    continue
                    
                filepath = os.path.join(root, file)
                
                # Determine Entity ID (Subject for solo, Pair for together)
                match_solo = re.search(r'(solo\d+)', folder_name)
                match_pair = re.search(r'(pair\d+)', folder_name)
                
                if match_solo:
                    entity_id = match_solo.group(1)
                elif match_pair:
                    entity_id = match_pair.group(1)
                else:
                    continue # Skip unrecognized folder structures
                
                entity_files[entity_id].append(filepath)

    # Splitting logic
    all_entities = sorted(list(entity_files.keys()))
    np.random.seed(42)
    np.random.shuffle(all_entities)
    
    # train_split = int(len(all_entities) * 0.8)
    # val_split = int(len(all_entities) * 0.9)
    
    # train_entities = all_entities[:train_split]
    # val_entities = all_entities[train_split:val_split]
    # test_entities = all_entities[val_split:]

    train_split = int(len(all_entities) * 0.8)

    train_entities = all_entities[:train_split]
    test_entities = all_entities[train_split:]
    val_entities = all_entities[train_split:] # Use the same entities for validation as test
    
    print(f"Found {len(all_entities)} unique entities (solo participants / pairs).")
    print(f"Train: {len(train_entities)} | Val: {len(val_entities)} | Test: {len(test_entities)}")

    # Setup output directories
    splits = {
        "train": (train_entities, os.path.join(final_root, "train")),
        "val":   (val_entities, os.path.join(final_root, "val")),
        "test":  (test_entities, os.path.join(final_root, "test"))
    }
    
    parameters = []
    for split_name, (entities, dump_folder) in splits.items():
        os.makedirs(dump_folder, exist_ok=True)
        for entity in entities:
            for filepath in entity_files[entity]:
                parameters.append([filepath, dump_folder, split_name, entity])

    print(f"Total .npz files queued for processing: {len(parameters)}")
    
    # Process files
    num_processes = 8
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_npz_file, parameters)

    # Aggregate and print statistics
    print("\n" + "="*50)
    print("PROCESSING COMPLETE - DATASET STATISTICS")
    print("="*50)
    
    stats = defaultdict(lambda: defaultdict(dict))
    
    for res in results:
        if res is not None:
            split_name, entity_id, vid_num, n_segments = res
            stats[split_name][entity_id][vid_num] = n_segments
            
    for split_name in ["train", "val", "test"]:
        print(f"\n### {split_name.upper()} SPLIT ###")
        total_split_segments = 0
        
        for entity in sorted(stats[split_name].keys()):
            entity_segments = 0
            vid_summaries = []
            
            for vid in sorted(stats[split_name][entity].keys()):
                n_seg = stats[split_name][entity][vid]
                entity_segments += n_seg
                vid_summaries.append(f"Vid {vid}: {n_seg}")
                
            total_split_segments += entity_segments
            print(f"  * {entity} (Total Trials: {entity_segments}) -> " + " | ".join(vid_summaries))
            
        print(f"  --> TOTAL TRIALS IN {split_name.upper()}: {total_split_segments}")

if __name__ == "__main__":
    main()