import os
import pickle
import re 
from multiprocessing import Pool
import numpy as np
import mne
from dotenv import load_dotenv
load_dotenv()
import shutil
from collections import defaultdict
from sklearn.model_selection import KFold

# It's good practice to keep a standard channel order for model compatibility.
# You should verify that these channels exist in your preprocessed data.
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
    Processes a single MNE epoch file. It iterates through each epoch,
    determines its condition (e.g., 'T1P' or 'T1Pn'), assigns a label,
    and saves each epoch individually as a pickle file.
    """
    filepath, dump_folder, chOrder_standard = params
    
    print("Processing", os.path.basename(filepath))
    try:
        # Load the pre-epoched data file
        epochs = mne.read_epochs(filepath, preload=True, verbose=False)
        epochs.crop(tmax=3.5)

        # Resample to 200 Hz if not already
        if epochs.info['sfreq'] != 200:
            epochs.resample(200, n_jobs=1)

        # Optional but recommended: reorder channels for consistency.
        if chOrder_standard is not None and len(chOrder_standard) == len(epochs.ch_names):
            if all(ch in epochs.ch_names for ch in chOrder_standard):
                 epochs.reorder_channels(chOrder_standard)
            else:
                print(f"Warning: Not all standard channels found in {filepath}. Skipping reorder.")

        # Create a reverse mapping from event ID (int) to condition name (str)
        # e.g., {1: 'T1P', 2: 'T1Pn'} -> {'T1P': 1, 'T1Pn': 2}
        rev_event_id = {v: k for k, v in epochs.event_id.items()}
        
        # Get all epoch data at once for efficiency
        data = epochs.get_data(units='uV')
        
        base_filename = os.path.basename(filepath).split('-epo.fif')[0]

        # Loop through each individual epoch contained in the file
        for i in range(len(epochs)):
            # Get the integer event code for this specific epoch
            event_code = epochs.events[i, 2]
            # Find the corresponding condition name (e.g., 'T1P', 'T1Pn')
            condition_name = rev_event_id[event_code]
            
            # Determine the label based on the condition name
            # Class 0 for no-feedback (ends with 'n'), Class 1 for feedback
            label = 0 if condition_name.endswith('n') else 1
            
            # Make the output filename more descriptive
            dump_path = os.path.join(
                dump_folder, f"{base_filename}_condition_{condition_name}_epoch{i}.pkl"
            )
            
            # The data for this single epoch is at index i
            epoch_data = data[i, :, :]
            
            pickle.dump(
                {"X": epoch_data, "y": label},
                open(dump_path, "wb"),
            )

    except Exception as e:
        print(f"ERROR processing file {filepath}: {e}")
        with open("my-dataset-process-error-files.txt", "a") as f:
            f.write(filepath + "\n")

def get_triad_from_filename(filename):
    """
    Extracts the triad ID from a filename like '301A_FG_...-epo.fif'.
    It captures the number at the start of the string.
    """
    match = re.search(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

if __name__ == "__main__":
    # Set path to your dataset folder (containing the 92 .fif files)
    raw_dataset_path = os.getenv("FG_DATASET_PATH") + "preprocessed/"
    
    # Group all epoch files by their triad ID
    all_files = [f for f in os.listdir(raw_dataset_path) if f.endswith("-epo.fif")]
    
    triad_files = defaultdict(list)
    for f in all_files:
        triad_id = get_triad_from_filename(f)
        if triad_id:
            triad_files[triad_id].append(f)
        else:
            print(f"Warning: Could not find triad ID for file {f}")

    # Split by triad ID for train, val, test
    # Convert to numpy array for KFold indexing
    all_triad_ids = np.array(sorted(triad_files.keys()))
    np.random.seed(42) # for reproducibility
    np.random.shuffle(all_triad_ids)

    # Split triads into 10 folds
    kf = KFold(n_splits=10)
    folds = []
    for _, test_idx in kf.split(all_triad_ids):
        folds.append(all_triad_ids[test_idx])

    print(f"Total triads: {len(all_triad_ids)}")

    # Set your processed data path
    final_root = os.getenv("FG_DATASET_PATH")
    base_dump_dir = os.path.join(final_root, "processed_cv")

    # Prepare parameters for the processing function
    parameters = []
    
    def prepare_params(file_list, dump_folder):
        params = []
        for f in file_list:
            filepath = os.path.join(raw_dataset_path, f)
            # The label is now determined inside the processing function
            params.append([filepath, dump_folder, standard_channels])
        return params

    # Create fold_0 to fold_9 and map files
    for fold_idx, fold_triads in enumerate(folds):
        fold_dump_folder = os.path.join(base_dump_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dump_folder, exist_ok=True)
        
        fold_files = [f for tid in fold_triads for f in triad_files[tid]]
        print(f"Fold {fold_idx} triads: {fold_triads}")
        
        parameters.extend(prepare_params(fold_files, fold_dump_folder))
    # Use a number of processes suitable for your machine
    with Pool(processes=8) as pool:
        pool.map(process_epoch_file, parameters)

    print("Processing complete.")