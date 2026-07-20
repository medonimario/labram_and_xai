import os
import pickle
import re
from multiprocessing import Pool
import numpy as np
import mne
from dotenv import load_dotenv
from collections import defaultdict
from sklearn.model_selection import KFold

# Load environment variables from .env file
load_dotenv()

# Standard channel order for model compatibility.
# Verify that these channels exist in your preprocessed data.
standard_channels = ['Fp1','Fpz','Fp2',
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

def process_epoch_file(params):
    """
    Processes a single EEGLAB epoch file (.set).
    Loads a 6s epoch, resamples to 200Hz, and saves two 
    overlapping 4s segments (0-4s and 2-6s).
    """
    filepath, dump_folder, chOrder_standard, label = params
    print(f"Processing {os.path.basename(filepath)} with label {label}")

    try:
        # --- Define segment parameters ---
        sfreq = 200.0
        total_len_s = 6.0
        segment_len_s = 4.0
        overlap_len_s = 2.0
        
        expected_samples_total = int(total_len_s * sfreq) # 6s * 200Hz = 1200
        segment_len_samples = int(segment_len_s * sfreq)   # 4s * 200Hz = 800
        step_size_samples = int((segment_len_s - overlap_len_s) * sfreq) # (4s-2s)*200Hz = 400

        # Load the pre-epoched EEGLAB data file
        epochs = mne.io.read_epochs_eeglab(filepath, verbose=False)
        epochs.set_eeg_reference('average', projection=True)
        
        # We want to load the full 6s segment first.

        # Resample to 200 Hz if not already
        if epochs.info['sfreq'] != sfreq:
            epochs.resample(sfreq, n_jobs=1, verbose=False)

        # Optional but recommended: reorder channels for consistency.
        if chOrder_standard and all(ch in epochs.ch_names for ch in chOrder_standard):
            epochs.reorder_channels(chOrder_standard)
        else:
            print(f"Warning: Not all standard channels found in {filepath}. Skipping reorder.")
        
        # Get all epoch data at once for efficiency
        data = epochs.get_data(units='uV')
        
        # --- Check shape of the 6s data ---
        if data.shape[2] != expected_samples_total:
            print(f"ERROR: File {filepath} data shape is {data.shape}, but expected {expected_samples_total} samples (6s @ 200Hz). Skipping file.")
            with open("my-dataset-process-error-files.txt", "a") as f:
                f.write(f"{filepath} - Expected {expected_samples_total} samples, got {data.shape[2]}\n")
            return # Skip this file
        
        print(f"Loaded 6s data shape for {os.path.basename(filepath)}: {data.shape}")
        base_filename = os.path.basename(filepath).split('.set')[0]

        # Loop through each individual 6s epoch contained in the file
        for i in range(len(epochs)):
            # The full 6s epoch data
            full_epoch_data = data[i, :, :] # Shape: (n_chans, 1200)

            # --- Create overlapping segments ---
            start_indices = [0, step_size_samples] # [0, 400]
            
            for seg_num, start_idx in enumerate(start_indices):
                end_idx = start_idx + segment_len_samples # 0+800=800; 400+800=1200
                
                # Crop the segment from the numpy array
                segment_data = full_epoch_data[:, start_idx:end_idx]

                # --- NEW: Check shape of the 4s segment ---
                expected_seg_shape = (full_epoch_data.shape[0], segment_len_samples)
                if segment_data.shape != expected_seg_shape:
                    print(f"ERROR: Segment {seg_num} for epoch {i} in {filepath} has wrong shape: {segment_data.shape}. Expected {expected_seg_shape}. Skipping segment.")
                    continue # Skip this segment
                
                # --- Modified dump_path to distinguish segments ---
                dump_path = os.path.join(
                    dump_folder, f"{base_filename}_epoch{i}_seg{seg_num}.pkl"
                )
                
                # The data for this single 4s segment
                pickle.dump(
                    {"X": segment_data, "y": label},
                    open(dump_path, "wb"),
                )

    except Exception as e:
        print(f"ERROR processing file {filepath}: {e}")
        with open("my-dataset-process-error-files.txt", "a") as f:
            f.write(f"{filepath} - {e}\n")

def get_dyad_from_filename(filename):
    """
    Extracts the dyad ID from a filename like 's_103_Coordination.set' or 's_203_Solo.set'.
    It captures the last two digits of the subject number as the dyad identifier.
    """
    match = re.search(r's_\d(\d{2})_', filename)
    if match:
        return int(match.group(1))
    return None

if __name__ == "__main__":
    # Set path to your dataset folder (containing the .set files)
    raw_dataset_path = os.getenv("MG_DATASET_PATH") + "preprocessed/"

    # Filter for .set files and only include 'Solo' or 'Spontaneous' conditions
    all_files = [
        f for f in os.listdir(raw_dataset_path) 
        if f.endswith(".set") and ('Solo' in f or 'Spontaneous' in f)
    ]
    
    # Group all relevant files by their Dyad ID
    dyad_files = defaultdict(list)
    for f in all_files:
        dyad_id = get_dyad_from_filename(f)
        if dyad_id is not None:
            dyad_files[dyad_id].append(f)
        else:
            print(f"Warning: Could not find structural dyad ID for file {f}")

    # Split by Dyad ID for train, val, test
    # Convert to numpy array for KFold indexing
    all_dyad_ids = np.array(sorted(dyad_files.keys()))
    np.random.seed(42) # for reproducibility
    np.random.shuffle(all_dyad_ids)

    # Split dyads into 10 folds
    kf = KFold(n_splits=10)
    folds = []
    for _, test_idx in kf.split(all_dyad_ids):
        folds.append(all_dyad_ids[test_idx])

    print(f"Total unique dyads: {len(all_dyad_ids)}")

    # Set your processed data path
    final_root = os.getenv("MG_DATASET_PATH") # Make sure to set this in your .env file
    base_dump_dir = os.path.join(final_root, "processed_solo-spont_cv")

    # Prepare parameters for the processing function
    parameters = []
    
    def prepare_params(file_list, dump_folder):
        params = []
        for f in file_list:
            filepath = os.path.join(raw_dataset_path, f)
            # Determine label from filename: 0 for Solo, 1 for Spontaneous
            label = 1 if 'Spontaneous' in f else 0
            params.append([filepath, dump_folder, standard_channels, label])
        return params

    # Create fold_0 to fold_9 and map files
    for fold_idx, fold_dyads in enumerate(folds):
        fold_dump_folder = os.path.join(base_dump_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dump_folder, exist_ok=True)
        
        fold_files = [f for did in fold_dyads for f in dyad_files[did]]
        print(f"Fold {fold_idx} dyads: {fold_dyads}")
        
        parameters.extend(prepare_params(fold_files, fold_dump_folder))

    # Use a number of processes suitable for your machine
    # Set to 1 if you encounter issues, to debug in serial mode.
    num_processes = 8
    with Pool(processes=num_processes) as pool:
        pool.map(process_epoch_file, parameters)

    print("Processing complete.")