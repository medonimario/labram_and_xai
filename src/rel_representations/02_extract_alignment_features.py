import os
import json
import pickle
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from einops import rearrange

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../labram_ft")))

import utils
from run_class_finetuning import get_models

class CirclingPklDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with open(path, 'rb') as f:
            data = pickle.load(f)
        # Expected X shape: (64, 800)
        eeg_tensor = torch.from_numpy(data['X']).float()
        label = data.get('y', -1)
        return eeg_tensor, label, path

def extract_features(model, dataloader, input_chans, target_layers, device):
    """
    Extracts pooled activations across all target layers in mini-batches.
    Returns: dict {layer_id: np.ndarray [num_samples, dim]} and metadata list.
    """
    model.eval()
    layer_acts = {l: [] for l in target_layers}
    labels_list = []
    paths_list = []

    with torch.no_grad():
        for eeg_batch, lbls, paths in tqdm(dataloader, desc="Extracting features"):
            # Preprocessing matches training: B N (A T) -> B N A T, / 100
            eeg_batch = rearrange(eeg_batch, 'B N (A T) -> B N A T', T=200)
            eeg_batch = eeg_batch.to(device) / 100.0

            # Forward pass through intermediate bottlenecks
            outputs = model.forward_intermediate(
                eeg_batch,
                layer_id=target_layers,
                norm_output=True,
                input_chans=input_chans
            )

            for i, l_id in enumerate(target_layers):
                act_tensor = outputs[i].cpu().numpy() # [B, 200]
                layer_acts[l_id].append(act_tensor)

            labels_list.extend(lbls.numpy().tolist())
            paths_list.extend(paths)

    # Concatenate batches
    for l_id in target_layers:
        layer_acts[l_id] = np.concatenate(layer_acts[l_id], axis=0)

    return layer_acts, np.array(labels_list), paths_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_a", type=str, required=True, help="Path to fold A .pth")
    parser.add_argument("--checkpoint_b", type=str, required=True, help="Path to fold B .pth")
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.manifest_path, "r") as f:
        manifest = json.load(f)

    anchor_files = manifest["anchors"]["files"]
    s_eval_files = manifest["s_eval"]["files"]

    anchor_loader = DataLoader(CirclingPklDataset(anchor_files), batch_size=args.batch_size, shuffle=False)
    s_eval_loader = DataLoader(CirclingPklDataset(s_eval_files), batch_size=args.batch_size, shuffle=False)

    target_layers = list(range(12)) # Layers 0 to 11

    # Channels definition from your code
    ch_names = ['FP1','FPZ','FP2','AF7','AF3','AFZ','AF4','AF8',
                'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
                'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
                'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
                'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
                'P9','P7','P5','P3','P1','PZ','P2','P4','P6','P8','P10',
                'PO7','PO3','POZ','PO4','PO8','O1','OZ','O2','IZ']
    input_chans = utils.get_input_chans(ch_names)

    checkpoints = {
        "model_a": args.checkpoint_a,
        "model_b": args.checkpoint_b
    }

    for model_name, ckpt_path in checkpoints.items():
        print(f"\n--- Processing {model_name} from {ckpt_path} ---")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model = get_models(ckpt['args'])
        state_dict = ckpt.get('model', ckpt.get('module', ckpt))
        utils.load_state_dict(model, state_dict)
        model.to(device)

        # 1. Anchors
        print("Extracting anchor features...")
        anch_acts, anch_y, anch_paths = extract_features(model, anchor_loader, input_chans, target_layers, device)
        
        # 2. Evaluation Set
        print("Extracting S_eval features...")
        eval_acts, eval_y, eval_paths = extract_features(model, s_eval_loader, input_chans, target_layers, device)

        # Save to disk
        out_file = os.path.join(args.output_dir, f"{model_name}_features.pkl")
        with open(out_file, "wb") as f:
            pickle.dump({
                "anchor_activations": anch_acts, # dict: {0: (K, 200), ... 11: (K, 200)}
                "eval_activations": eval_acts,     # dict: {0: (N, 200), ... 11: (N, 200)}
                "anchor_meta": {"y": anch_y, "paths": anch_paths},
                "eval_meta": {"y": eval_y, "paths": eval_paths}
            }, f)
        print(f"Saved extracted features to {out_file}")

if __name__ == "__main__":
    main()