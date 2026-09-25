import os
import pickle
import re
from multiprocessing import Pool
from collections import defaultdict
import numpy as np
import mne
from dotenv import load_dotenv
from sklearn.model_selection import KFold

load_dotenv()

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

def get_bad_ranges(bad_indices_global):
    """Converts 6s global segment indices to per-trial (tmin, tmax) ranges."""
    bad_ranges = defaultdict(list)
    for global_idx in bad_indices_global:
        trial_idx = global_idx // 3
        segment_in_trial = global_idx % 3
        bad_start = segment_in_trial * 6.0
        bad_end = bad_start + 6.0
        bad_ranges[trial_idx].append((bad_start, bad_end))
    return bad_ranges

def standardize_epoch(raw_data, info, chOrder_standard):
    """Resamples to 200Hz and reorders channels standardly."""
    temp_obj = mne.EpochsArray(raw_data[np.newaxis, :, :], info, verbose=False)
    if temp_obj.info['sfreq'] != 200:
        temp_obj.resample(200, n_jobs=1, verbose=False)
    if chOrder_standard:
        temp_obj.reorder_channels(chOrder_standard)
    return temp_obj.get_data()[0]

def process_dyad(params):
    """
    Processes a dyad pair together, applying dyadic artifact rejection
    (union of bad ranges) to produce perfectly synchronized segments.
    """
    pair_id, file_pair, dump_folder, chOrder_standard, bad_epochs_dict = params
    
    if len(file_pair) != 2:
        print(f"Warning: Pair {pair_id} does not have exactly 2 files (found {len(file_pair)}). Skipping.")
        return

    # Deterministic ordering so SubA and SubB are consistently assigned
    file_pair = sorted(file_pair)
    file_a, file_b = file_pair[0], file_pair[1]
    
    try:
        epochs_a = mne.read_epochs(file_a, preload=True, verbose=False)
        epochs_b = mne.read_epochs(file_b, preload=True, verbose=False)

        practice_trial_cut = 12 if str(pair_id) == '2030' else 6
        epochs_a = epochs_a.crop(tmin=2, verbose=False)[practice_trial_cut:]
        epochs_b = epochs_b.crop(tmin=2, verbose=False)[practice_trial_cut:]

        conditions = {'open': 1, 'closed1': 0}

        for condition, label in conditions.items():
            if condition not in epochs_a.event_id or condition not in epochs_b.event_id:
                continue

            cond_a = epochs_a[condition]
            cond_b = epochs_b[condition]

            # Ensure both participants have the exact same number of trials
            n_trials = min(len(cond_a), len(cond_b))
            if len(cond_a) != len(cond_b):
                print(f"Warning: Trial count mismatch in pair {pair_id} for {condition}: "
                      f"{len(cond_a)} vs {len(cond_b)}. Truncating to {n_trials}.")

            # Retrieve bad ranges.
            # If your dictionary keys are per-subject, look them up per subject.
            # If your dictionary keys are per-pair, retrieve once.
            bad_key = f"{pair_id}_{condition}"
            bad_ranges = get_bad_ranges(bad_epochs_dict.get(bad_key, []))

            sfreq = epochs_a.info['sfreq']
            data_a = cond_a.get_data(units='uV')[:n_trials]
            data_b = cond_b.get_data(units='uV')[:n_trials]

            segm_len_samples = int(4 * sfreq)
            overlap_samples = int(2 * sfreq)
            step_size_samples = segm_len_samples - overlap_samples

            saved_segments = 0
            for trial_idx in range(n_trials):
                trial_data_a = data_a[trial_idx]
                trial_data_b = data_b[trial_idx]
                bad_ranges_trial = bad_ranges.get(trial_idx, [])
                n_samples = trial_data_a.shape[1]

                for win_idx, start_sample in enumerate(range(0, n_samples - segm_len_samples + 1, step_size_samples)):
                    end_sample = start_sample + segm_len_samples
                    tmin = start_sample / sfreq
                    tmax = end_sample / sfreq

                    # Dyadic check: bad in either subject drops the segment for both
                    is_bad = any(tmin < b_end and tmax > b_start for b_start, b_end in bad_ranges_trial)
                    if is_bad:
                        continue

                    # Slice matching clean segments
                    seg_a = trial_data_a[:, start_sample:end_sample]
                    seg_b = trial_data_b[:, start_sample:end_sample]

                    final_a = standardize_epoch(seg_a, epochs_a.info, chOrder_standard)
                    final_b = standardize_epoch(seg_b, epochs_b.info, chOrder_standard)

                    # Save paired data together with explicit trial/window coordinates
                    dump_path = os.path.join(
                        dump_folder,
                        f"pair{pair_id}_cond_{condition}_trial{trial_idx}_win{win_idx}.pkl"
                    )

                    pickle.dump(
                        {
                            "X_a": final_a,
                            "X_b": final_b,
                            "y": label,
                            "pair_id": pair_id,
                            "trial_idx": trial_idx,
                            "win_idx": win_idx
                        },
                        open(dump_path, "wb")
                    )
                    saved_segments += 1

            print(f"Pair {pair_id} [{condition}]: Saved {saved_segments} aligned segment pairs.")

    except Exception as e:
        print(f"ERROR processing pair {pair_id}: {e}")
        with open("my-dataset-process-error-files.txt", "a") as f:
            f.write(f"Pair {pair_id}: {file_a}, {file_b} -> {e}\n")

def get_pair_from_filename(filename):
    match = re.search(r'pair(\d+)', filename)
    return int(match.group(1)) if match else None

if __name__ == "__main__":
    base_path = os.getenv("CIRCLING_DATASET_PATH")
    raw_dataset_path = os.path.join(base_path, "preprocessed/")
    bad_epochs_pickle_path = os.path.join(base_path, "dict_with_short_bad_epochs.pickle")

    with open(bad_epochs_pickle_path, "rb") as f:
        bad_short_epochs = pickle.load(f)

    all_files = [f for f in os.listdir(raw_dataset_path) if f.endswith("_ave_epo.fif")]
    
    pair_files = defaultdict(list)
    for f in all_files:
        pair_id = get_pair_from_filename(f)
        if pair_id:
            pair_files[pair_id].append(os.path.join(raw_dataset_path, f))

    all_pair_ids = np.array(sorted(pair_files.keys()))
    np.random.seed(42)
    np.random.shuffle(all_pair_ids)

    # 10-fold split at the pair level
    kf = KFold(n_splits=10)
    folds = [all_pair_ids[test_idx] for _, test_idx in kf.split(all_pair_ids)]

    final_root = os.getenv("CIRCLING_PROCESSED_PATH", os.path.join(base_path, "processed_cv_cl/"))
    parameters = []

    for fold_idx, fold_pairs in enumerate(folds):
        fold_dump_folder = os.path.join(final_root, f"fold_{fold_idx}")
        os.makedirs(fold_dump_folder, exist_ok=True)
        
        for pid in fold_pairs:
            parameters.append([
                pid,
                pair_files[pid],
                fold_dump_folder,
                standard_channels,
                bad_short_epochs
            ])

    with Pool(processes=8) as pool:
        pool.map(process_dyad, parameters)

    print("Processing complete.")