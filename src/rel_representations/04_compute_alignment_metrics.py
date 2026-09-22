import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

def compute_pairwise_cosine(feat_a: torch.Tensor, feat_b: torch.Tensor) -> np.ndarray:
    """Computes direct cosine similarity between identical sample rows."""
    u = F.normalize(feat_a, p=2, dim=-1)
    v = F.normalize(feat_b, p=2, dim=-1)
    cos = (u * v).sum(dim=-1)
    return cos.cpu().numpy()

def compute_mrr(feat_a: torch.Tensor, feat_b: torch.Tensor, batch_size: int = 512) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) where query sample i in space A 
    searches for target sample i in space B across all N candidates.
    """
    u = F.normalize(feat_a, p=2, dim=-1)
    v = F.normalize(feat_b, p=2, dim=-1)
    n_samples = u.shape[0]
    ranks = []

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_u = u[start_idx:end_idx] # [B, D]

        # Cosine similarity matrix between batch queries and all target samples [B, N]
        sim_matrix = torch.matmul(batch_u, v.transpose(0, 1))

        # True sample indices for this batch
        true_indices = torch.arange(start_idx, end_idx, device=u.device).unsqueeze(1) # [B, 1]
        true_scores = sim_matrix.gather(1, true_indices) # [B, 1]

        # Rank = 1 + number of samples with strictly higher similarity
        batch_ranks = (sim_matrix > true_scores).sum(dim=1) + 1
        ranks.extend(batch_ranks.cpu().numpy().tolist())

    reciprocal_ranks = 1.0 / np.array(ranks)
    return float(np.mean(reciprocal_ranks))

def compute_jaccard_knn(feat_a: torch.Tensor, feat_b: torch.Tensor, k: int = 10, batch_size: int = 512) -> float:
    """
    Computes mean Jaccard overlap between the k-NN neighborhood in space A 
    and the k-NN neighborhood in space B.
    """
    u = F.normalize(feat_a, p=2, dim=-1)
    v = F.normalize(feat_b, p=2, dim=-1)
    n_samples = u.shape[0]
    jaccards = []

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        
        sim_a = torch.matmul(u[start_idx:end_idx], u.transpose(0, 1)) # [B, N]
        sim_b = torch.matmul(v[start_idx:end_idx], v.transpose(0, 1)) # [B, N]

        # Exclude self-similarity by setting diagonal entries to -infinity
        for b_i, global_i in enumerate(range(start_idx, end_idx)):
            sim_a[b_i, global_i] = -float('inf')
            sim_b[b_i, global_i] = -float('inf')

        # Retrieve top-k nearest neighbor indices
        _, knn_a = torch.topk(sim_a, k=k, dim=1) # [B, k]
        _, knn_b = torch.topk(sim_b, k=k, dim=1) # [B, k]

        knn_a_np = knn_a.cpu().numpy()
        knn_b_np = knn_b.cpu().numpy()

        for b_i in range(len(knn_a_np)):
            set_a = set(knn_a_np[b_i])
            set_b = set(knn_b_np[b_i])
            inter = len(set_a.intersection(set_b))
            union = len(set_a.union(set_b))
            jaccards.append(inter / union if union > 0 else 0.0)

    return float(np.mean(jaccards))

def evaluate_layer(feat_a_np: np.ndarray, feat_b_np: np.ndarray, labels: np.ndarray, device: torch.device, k: int = 10):
    """Computes Cosine, MRR, and Jaccard overall and split by condition."""
    feat_a = torch.from_numpy(feat_a_np).float().to(device)
    feat_b = torch.from_numpy(feat_b_np).float().to(device)

    # 1. Overall metrics
    cosines = compute_pairwise_cosine(feat_a, feat_b)
    mrr = compute_mrr(feat_a, feat_b)
    jaccard = compute_jaccard_knn(feat_a, feat_b, k=k)

    results = {
        "cosine_mean": float(np.mean(cosines)),
        "cosine_std": float(np.std(cosines)),
        "mrr": mrr,
        "jaccard": jaccard
    }

    # 2. Condition-specific metrics (open=1, closed1=0)
    for cond_name, cond_val in [("closed", 0), ("open", 1)]:
        mask = (labels == cond_val)
        if np.any(mask):
            results[f"cosine_{cond_name}"] = float(np.mean(cosines[mask]))
            feat_a_sub = feat_a[mask]
            feat_b_sub = feat_b[mask]
            results[f"mrr_{cond_name}"] = compute_mrr(feat_a_sub, feat_b_sub)
            results[f"jaccard_{cond_name}"] = compute_jaccard_knn(feat_a_sub, feat_b_sub, k=k)

    return results

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Compute alignment metrics across layers.")
    parser.add_argument("--model_a_proj", type=str, required=True, help="Path to model_a_relative.pkl")
    parser.add_argument("--model_b_proj", type=str, required=True, help="Path to model_b_relative.pkl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--knn_k", type=int, default=10, help="k for Jaccard similarity")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running metric evaluation on device: {device}")

    with open(args.model_a_proj, "rb") as f:
        proj_a = pickle.load(f)
    with open(args.model_b_proj, "rb") as f:
        proj_b = pickle.load(f)

    labels = np.array(proj_a["eval_meta"]["y"])
    layers = sorted(proj_a["relative_activations"].keys())
    records = []

    print(f"\nComputing metrics across {len(layers)} layers...")
    for l_id in layers:
        print(f"--- Layer {l_id} ---")

        # 1. Absolute Representations (Centered & L2-normalized)
        abs_a = proj_a["abs_norm_activations"][l_id]
        abs_b = proj_b["abs_norm_activations"][l_id]
        res_abs = evaluate_layer(abs_a, abs_b, labels, device, k=args.knn_k)
        res_abs.update({"layer": l_id, "space": "Absolute"})
        records.append(res_abs)
        print(f"  [Absolute] Cosine: {res_abs['cosine_mean']:.3f} | MRR: {res_abs['mrr']:.3f} | Jaccard: {res_abs['jaccard']:.3f}")

        # 2. Relative Representations
        rel_a = proj_a["relative_activations"][l_id]
        rel_b = proj_b["relative_activations"][l_id]
        res_rel = evaluate_layer(rel_a, rel_b, labels, device, k=args.knn_k)
        res_rel.update({"layer": l_id, "space": "Relative"})
        records.append(res_rel)
        print(f"  [Relative] Cosine: {res_rel['cosine_mean']:.3f} | MRR: {res_rel['mrr']:.3f} | Jaccard: {res_rel['jaccard']:.3f}")

    df_results = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, "alignment_metrics_summary.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nAll metrics successfully saved to: {csv_path}")

if __name__ == "__main__":
    main()