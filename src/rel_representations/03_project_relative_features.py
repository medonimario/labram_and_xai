import os
import pickle
import argparse
import numpy as np
import torch
from relative_representation import RelativeRepresentation

def project_model_features(features_dict, similarity="cosine", center=True):
    """
    Transforms feature activations across all layers into relative coordinates.
    """
    anchor_acts = features_dict["anchor_activations"] # {layer_id: [K, 200]}
    eval_acts = features_dict["eval_activations"]     # {layer_id: [N, 200]}
    
    relative_layers = {}
    normalized_abs_layers = {}

    for layer_id in sorted(anchor_acts.keys()):
        A_l = anchor_acts[layer_id]
        X_l = eval_acts[layer_id]

        # Initialize projector with layer-specific anchors
        projector = RelativeRepresentation(
            anchors=A_l,
            similarity=similarity,
            center=center
        )

        # 1. Project evaluation samples to relative space
        R_l = projector(X_l) # Shape: [N, K]
        relative_layers[layer_id] = R_l

        # 2. Compute centered and unit-normalized absolute embeddings for fair baseline
        if center:
            mu = np.mean(A_l, axis=0, keepdims=True)
            X_centered = X_l - mu
        else:
            X_centered = X_l
        
        norms = np.linalg.norm(X_centered, axis=-1, keepdims=True) + 1e-8
        normalized_abs_layers[layer_id] = X_centered / norms

    return {
        "relative_activations": relative_layers,
        "abs_norm_activations": normalized_abs_layers,
        "eval_meta": features_dict["eval_meta"]
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Project extracted features into relative space.")
    parser.add_argument("--features_a", type=str, required=True, help="Path to model_a_features.pkl")
    parser.add_argument("--features_b", type=str, required=True, help="Path to model_b_features.pkl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--similarity", type=str, default="cosine", choices=["cosine", "euclidean", "inner_product"])
    parser.add_argument("--no_center", action="store_true", help="Disable mean centering before projection")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    center = not args.no_center

    print(f"Loading extracted features...")
    with open(args.features_a, "rb") as f:
        feat_a = pickle.load(f)
    with open(args.features_b, "rb") as f:
        feat_b = pickle.load(f)

    print(f"Projecting Model A (similarity={args.similarity}, center={center})...")
    proj_a = project_model_features(feat_a, similarity=args.similarity, center=center)

    print(f"Projecting Model B (similarity={args.similarity}, center={center})...")
    proj_b = project_model_features(feat_b, similarity=args.similarity, center=center)

    out_a = os.path.join(args.output_dir, "model_a_relative.pkl")
    out_b = os.path.join(args.output_dir, "model_b_relative.pkl")

    with open(out_a, "wb") as f:
        pickle.dump(proj_a, f)
    with open(out_b, "wb") as f:
        pickle.dump(proj_b, f)

    print(f"Relative projection complete:")
    print(f"  Model A -> {out_a}")
    print(f"  Model B -> {out_b}")

if __name__ == "__main__":
    main()