import os
import re
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

def parse_metadata_from_path(filepath):
    """
    Extracts pair_id, subject_id, condition, and segment_idx from filename.
    Handles naming formats like:
      pair2001_p1_condition_open_segment0.pkl
      pair2001_1_condition_closed1_segment3.pkl
      pair2001A_condition_open_segment0.pkl
    """
    fname = os.path.basename(filepath)
    
    # 1. Pair ID
    pair_match = re.search(r'pair(\d+)', fname, re.IGNORECASE)
    pair_id = pair_match.group(1) if pair_match else "unknown"

    # 2. Condition
    cond_match = re.search(r'condition_([a-zA-Z0-9]+)', fname)
    condition = cond_match.group(1) if cond_match else "unknown"

    # 3. Segment Index
    seg_match = re.search(r'segment(\d+)', fname)
    seg_idx = int(seg_match.group(1)) if seg_match else -1

    # 4. Subject ID (isolates token between pair\d+ and _condition)
    sub_match = re.search(r'pair\d+[_-]?([a-zA-Z0-9]+?)(?:_condition|_ave|\.pkl)', fname, re.IGNORECASE)
    if sub_match and sub_match.group(1).strip('_'):
        subject_id = sub_match.group(1).strip('_')
    else:
        fallback = re.search(r'([pP]\d+|sub(?:ject)?\d+|_\d+_)', fname)
        subject_id = fallback.group(1).strip('_') if fallback else "p1"

    return {
        "filepath": filepath,
        "pair_id": pair_id,
        "subject_id": subject_id,
        "condition": condition,
        "segment_idx": seg_idx
    }

def compute_rbf_mmd(X, Y, gamma=None):
    """
    Computes unbiased MMD^2 with RBF kernel between point clouds X [N1, D] and Y [N2, D].
    """
    nx, ny = X.shape[0], Y.shape[0]
    if nx < 2 or ny < 2:
        return np.nan

    XX = np.sum(X**2, axis=1, keepdims=True) + np.sum(X**2, axis=1, keepdims=True).T - 2 * np.dot(X, X.T)
    YY = np.sum(Y**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1, keepdims=True).T - 2 * np.dot(Y, Y.T)
    XY = np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1, keepdims=True).T - 2 * np.dot(X, Y.T)

    XX = np.maximum(XX, 0.0)
    YY = np.maximum(YY, 0.0)
    XY = np.maximum(XY, 0.0)

    if gamma is None:
        median_dist = np.median(XY)
        gamma = 1.0 / (2.0 * median_dist + 1e-8) if median_dist > 0 else 0.01

    K_XX = np.exp(-gamma * XX)
    K_YY = np.exp(-gamma * YY)
    K_XY = np.exp(-gamma * XY)

    mmd2 = (np.sum(K_XX) - np.trace(K_XX)) / (nx * (nx - 1) + 1e-8) \
         + (np.sum(K_YY) - np.trace(K_YY)) / (ny * (ny - 1) + 1e-8) \
         - 2.0 * np.mean(K_XY)
         
    return float(np.maximum(mmd2, 0.0))

def compute_cross_subject_probe(X1, y1, X2, y2):
    """
    Trains a linear probe on Subject 1 and tests zero-shot on Subject 2 (and vice versa).
    Returns mean cross-subject balanced accuracy and within-subject CV accuracy.
    """
    u1, c1 = np.unique(y1, return_counts=True)
    u2, c2 = np.unique(y2, return_counts=True)

    # Require at least 2 classes and minimum 5 samples per class
    if len(u1) < 2 or len(u2) < 2 or min(c1) < 5 or min(c2) < 5:
        return np.nan, np.nan

    try:
        # S1 -> S2 transfer
        clf1 = LogisticRegression(C=1.0, max_iter=400, solver='lbfgs', random_state=42)
        clf1.fit(X1, y1)
        acc_1to2 = balanced_accuracy_score(y2, clf1.predict(X2))

        # S2 -> S1 transfer
        clf2 = LogisticRegression(C=1.0, max_iter=400, solver='lbfgs', random_state=42)
        clf2.fit(X2, y2)
        acc_2to1 = balanced_accuracy_score(y1, clf2.predict(X1))

        cross_acc = float((acc_1to2 + acc_2to1) / 2.0)

        # Within-subject 3-fold CV as ceiling
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv1 = [balanced_accuracy_score(y1[te], clf1.fit(X1[tr], y1[tr]).predict(X1[te])) for tr, te in skf.split(X1, y1)]
        cv2 = [balanced_accuracy_score(y2[te], clf2.fit(X2[tr], y2[tr]).predict(X2[te])) for tr, te in skf.split(X2, y2)]
        within_acc = float((np.mean(cv1) + np.mean(cv2)) / 2.0)

        return cross_acc, within_acc
    except Exception:
        return np.nan, np.nan

def evaluate_subject_pair(feat_1, y_1, meta_1, feat_2, y_2, meta_2):
    """
    Evaluates alignment between Subject 1 and Subject 2 in a given space.
    """
    # 1. Condition-Centroid Cosine Similarity
    results = {}
    for cond_val, cond_name in [(1, "open"), (0, "closed")]:
        mask1 = (y_1 == cond_val)
        mask2 = (y_2 == cond_val)
        if np.sum(mask1) >= 2 and np.sum(mask2) >= 2:
            mu1 = np.mean(feat_1[mask1], axis=0, keepdims=True)
            mu2 = np.mean(feat_2[mask2], axis=0, keepdims=True)
            cos = np.dot(mu1, mu2.T) / (np.linalg.norm(mu1) * np.linalg.norm(mu2) + 1e-8)
            results[f"centroid_cosine_{cond_name}"] = float(cos[0, 0])
            results[f"mmd_{cond_name}"] = compute_rbf_mmd(feat_1[mask1], feat_2[mask2])
        else:
            results[f"centroid_cosine_{cond_name}"] = np.nan
            results[f"mmd_{cond_name}"] = np.nan

    # Cross-condition contrast (Sanity check: S1 open vs S2 closed)
    mask1_open = (y_1 == 1)
    mask2_closed = (y_2 == 0)
    if np.sum(mask1_open) >= 2 and np.sum(mask2_closed) >= 2:
        mu1_o = np.mean(feat_1[mask1_open], axis=0, keepdims=True)
        mu2_c = np.mean(feat_2[mask2_closed], axis=0, keepdims=True)
        cos_cross = np.dot(mu1_o, mu2_c.T) / (np.linalg.norm(mu1_o) * np.linalg.norm(mu2_c) + 1e-8)
        results["centroid_cosine_cross_cond"] = float(cos_cross[0, 0])
    else:
        results["centroid_cosine_cross_cond"] = np.nan

    # 2. Overall Manifold MMD
    results["mmd_overall"] = compute_rbf_mmd(feat_1, feat_2)

    # 3. Cross-Subject Functional Probing
    cross_acc, within_acc = compute_cross_subject_probe(feat_1, y_1, feat_2, y_2)
    results["probe_cross_acc"] = cross_acc
    results["probe_within_ceiling"] = within_acc

    # 4. Synchronous Segment Paired Similarity (where segment indices match)
    s1_dict = {(m["condition"], m["segment_idx"]): idx for idx, m in enumerate(meta_1) if m["segment_idx"] >= 0}
    s2_dict = {(m["condition"], m["segment_idx"]): idx for idx, m in enumerate(meta_2) if m["segment_idx"] >= 0}
    common_keys = set(s1_dict.keys()).intersection(set(s2_dict.keys()))

    if len(common_keys) >= 3:
        idx1 = [s1_dict[k] for k in common_keys]
        idx2 = [s2_dict[k] for k in common_keys]
        p1_sync = feat_1[idx1]
        p2_sync = feat_2[idx2]
        # Direct cosine along matching rows
        u = p1_sync / (np.linalg.norm(p1_sync, axis=-1, keepdims=True) + 1e-8)
        v = p2_sync / (np.linalg.norm(p2_sync, axis=-1, keepdims=True) + 1e-8)
        paired_cos = np.sum(u * v, axis=-1)
        results["sync_paired_cosine"] = float(np.mean(paired_cos))
        results["sync_paired_count"] = len(common_keys)
    else:
        results["sync_paired_cosine"] = np.nan
        results["sync_paired_count"] = 0

    return results

def main():
    parser = argparse.ArgumentParser(description="Idea 2: Dyadic Inter-Subject Representation Alignment.")
    parser.add_argument("--model_proj", type=str, required=True, help="Path to model_a_relative.pkl")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading representations from: {args.model_proj}")
    with open(args.model_proj, "rb") as f:
        proj_data = pickle.load(f)

    paths = proj_data["eval_meta"]["paths"]
    labels = np.array(proj_data["eval_meta"]["y"])
    layers = sorted(proj_data["relative_activations"].keys())

    # 1. Parse sample metadata
    metadata = [parse_metadata_from_path(p) for p in paths]

    # 2. Group sample indices by Pair ID and Subject ID
    dyad_members = defaultdict(lambda: defaultdict(list))
    for i, meta in enumerate(metadata):
        dyad_members[meta["pair_id"]][meta["subject_id"]].append(i)

    # Filter for valid dyads (at least 2 subjects with >= 10 samples each)
    valid_dyads = {}
    for p_id, sub_dict in dyad_members.items():
        qualifying = {s: idxs for s, idxs in sub_dict.items() if len(idxs) >= 10}
        if len(qualifying) >= 2:
            valid_dyads[p_id] = qualifying

    pair_ids = sorted(valid_dyads.keys())
    print(f"Found {len(pair_ids)} valid dyads with multiple participants: {pair_ids}")
    if len(pair_ids) < 2:
        print("Warning: Need at least 2 dyads to compute surrogate controls.")

    detailed_records = []

    print("\nComputing dyadic representation alignment across all layers...")
    for l_id in tqdm(layers, desc="Layers"):
        for space_name in ["Absolute", "Relative"]:
            if space_name == "Absolute":
                feats_all = proj_data["abs_norm_activations"][l_id]
            else:
                feats_all = proj_data["relative_activations"][l_id]

            # Normalize embeddings to unit norm
            norms = np.linalg.norm(feats_all, axis=-1, keepdims=True) + 1e-8
            feats_all_norm = feats_all / norms

            # --- A. REAL DYADS ---
            for p_idx, p_id in enumerate(pair_ids):
                subs = sorted(valid_dyads[p_id].keys())
                for s1, s2 in combinations(subs, 2):
                    idx1 = valid_dyads[p_id][s1]
                    idx2 = valid_dyads[p_id][s2]

                    res_real = evaluate_subject_pair(
                        feat_1=feats_all_norm[idx1],
                        y_1=labels[idx1],
                        meta_1=[metadata[i] for i in idx1],
                        feat_2=feats_all_norm[idx2],
                        y_2=labels[idx2],
                        meta_2=[metadata[i] for i in idx2]
                    )

                    res_real.update({
                        "layer": l_id,
                        "space": space_name,
                        "pair_id": p_id,
                        "dyad_type": "Real",
                        "subject_1": s1,
                        "subject_2": s2
                    })
                    detailed_records.append(res_real)

                    # --- B. SURROGATE DYADS (Interpersonal Control) ---
                    # Pair Subject 1 from dyad p with Subject 2 from a circular-shifted dyad
                    if len(pair_ids) > 1:
                        surr_p_id = pair_ids[(p_idx + 1) % len(pair_ids)]
                        surr_s2 = sorted(valid_dyads[surr_p_id].keys())[0]
                        idx_surr2 = valid_dyads[surr_p_id][surr_s2]

                        res_surr = evaluate_subject_pair(
                            feat_1=feats_all_norm[idx1],
                            y_1=labels[idx1],
                            meta_1=[metadata[i] for i in idx1],
                            feat_2=feats_all_norm[idx_surr2],
                            y_2=labels[idx_surr2],
                            meta_2=[metadata[i] for i in idx_surr2]
                        )

                        res_surr.update({
                            "layer": l_id,
                            "space": space_name,
                            "pair_id": f"{p_id}_x_{surr_p_id}",
                            "dyad_type": "Surrogate",
                            "subject_1": s1,
                            "subject_2": surr_s2
                        })
                        detailed_records.append(res_surr)

    # 3. Save Output DataFrames
    df_detailed = pd.DataFrame(detailed_records)
    detailed_csv = os.path.join(args.output_dir, "dyadic_per_pair_metrics.csv")
    df_detailed.to_csv(detailed_csv, index=False)

    # 4. Generate Aggregated Summary Table (Mean ± SE across pairs)
    numeric_cols = [c for c in df_detailed.columns if c not in ["layer", "space", "pair_id", "dyad_type", "subject_1", "subject_2"]]
    df_summary = df_detailed.groupby(["layer", "space", "dyad_type"])[numeric_cols].agg(['mean', 'std']).reset_index()

    summary_csv = os.path.join(args.output_dir, "dyadic_alignment_summary.csv")
    df_summary.to_csv(summary_csv, index=False)

    print(f"\nCompleted successfully!")
    print(f"  Per-dyad detailed records -> {detailed_csv}")
    print(f"  Aggregated summary table -> {summary_csv}")

if __name__ == "__main__":
    main()