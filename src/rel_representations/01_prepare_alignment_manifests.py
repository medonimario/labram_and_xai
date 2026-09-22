import os
import re
import json
import random
import pickle
import argparse
from glob import glob

def get_pair_from_filename(filename):
    match = re.search(r'pair(\d+)', filename)
    return int(match.group(1)) if match else None

def load_pkl_metadata(filepath):
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            return data.get('y', None)
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", type=str, required=True,
                        help="Path to processed_cv root containing fold_0 ... fold_9")
    parser.add_argument("--baseline_dir", type=str, required=True,
                        help="Directory containing clean resting-state baseline .pkl files")
    parser.add_argument("--fold_a", type=int, default=0)
    parser.add_argument("--fold_b", type=int, default=1)
    parser.add_argument("--num_baseline_anchors", type=int, default=150)
    parser.add_argument("--num_test_anchors_per_class", type=int, default=75) # 75 open + 75 closed = 150
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Collect all test samples from both folds
    dir_a = os.path.join(args.processed_root, f"fold_{args.fold_a}")
    dir_b = os.path.join(args.processed_root, f"fold_{args.fold_b}")
    
    files_a = sorted(glob(os.path.join(dir_a, "*.pkl")))
    files_b = sorted(glob(os.path.join(dir_b, "*.pkl")))
    
    combined_test_files = files_a + files_b
    print(f"Loaded {len(files_a)} files from Fold {args.fold_a} and {len(files_b)} files from Fold {args.fold_b}.")

    # 2. Collect & balance candidate test anchors
    open_candidates = []
    closed_candidates = []
    
    for fpath in combined_test_files:
        lbl = load_pkl_metadata(fpath)
        if lbl == 1:
            open_candidates.append(fpath)
        elif lbl == 0:
            closed_candidates.append(fpath)

    k_test_class = args.num_test_anchors_per_class
    test_anchors_open = random.sample(open_candidates, min(k_test_class, len(open_candidates)))
    test_anchors_closed = random.sample(closed_candidates, min(k_test_class, len(closed_candidates)))
    test_anchors = test_anchors_open + test_anchors_closed

    # 3. Collect resting baseline anchors
    baseline_files = sorted(glob(os.path.join(args.baseline_dir, "*.pkl")))
    k_base = min(args.num_baseline_anchors, len(baseline_files))
    baseline_anchors = random.sample(baseline_files, k_base)

    # 4. Final anchor set (Baseline + Balanced Test Anchors)
    final_anchors = baseline_anchors + test_anchors
    
    # 5. Define Evaluation Set (S_eval)
    # Exclude any files chosen as anchors from S_eval to avoid self-similarity trivialities
    anchor_set_lookup = set(final_anchors)
    s_eval_files = [f for f in combined_test_files if f not in anchor_set_lookup]

    manifest = {
        "fold_a": args.fold_a,
        "fold_b": args.fold_b,
        "anchors": {
            "total_count": len(final_anchors),
            "baseline_count": len(baseline_anchors),
            "test_open_count": len(test_anchors_open),
            "test_closed_count": len(test_anchors_closed),
            "files": final_anchors
        },
        "s_eval": {
            "total_count": len(s_eval_files),
            "files": s_eval_files
        }
    }

    manifest_path = os.path.join(args.output_dir, "alignment_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Manifest written to: {manifest_path}")
    print(f"Total Anchors: {len(final_anchors)} | Total S_eval samples: {len(s_eval_files)}")

if __name__ == "__main__":
    main()