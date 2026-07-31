import os
import sys
import argparse
import torch
import numpy as np
from torch.nn import functional as F 
from einops import rearrange

# Compute paths relative to this script's location
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))          
sys.path.append(os.path.abspath(os.path.join(current_dir, "../labram_ft"))) 

from labram_ft.run_class_finetuning import get_dataset, get_models
import utils

def extract_and_save_attention(checkpoint_dir, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint-best.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")

    print("--- Loading Checkpoint and Arguments ---")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args = checkpoint['args']
    args.use_attention_pooling = True 
    print(f"Dataset: {args.dataset}")
    
    # 1. Load Dataset
    _, test_dataset, _, ch_names, _ = get_dataset(args)
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 2. Load Model
    model = get_models(args)
    utils.load_state_dict(model, checkpoint.get('model', checkpoint.get('module', checkpoint)))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 3. Setup Forward Hook
    saved_attn_logits = []
    def hook_fn(module, input, output):
        saved_attn_logits.append(output.detach())
    handle = model.attention_pool.register_forward_hook(hook_fn)
    input_chans = utils.get_input_chans(ch_names)

    all_targets = []

    print("--- Extracting Attention Weights ---")
    with torch.no_grad():
        for batch in data_loader_test:
            EEG = batch[0].float().to(device) / 100
            EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200) 
            target = batch[-1].to(device)
            
            _ = model(EEG, input_chans=input_chans)
            print(f"Logits collected so far: {len(saved_attn_logits)}")
            all_targets.append(target.cpu().numpy())

    handle.remove()

    # 4. Process Logits into Softmax
    all_attn_logits = torch.cat(saved_attn_logits, dim=0)          
    all_attn_weights = F.softmax(all_attn_logits, dim=1).cpu().numpy()
    all_targets = np.concatenate(all_targets, axis=0)              

    B, N_A, _ = all_attn_weights.shape
    N = len(ch_names)       
    A = N_A // N            
    
    attn_maps = all_attn_weights.reshape(B, N, A)

    if all_targets.ndim > 1:
        all_targets = all_targets.squeeze()
        
    class_0_maps = attn_maps[all_targets == 0].mean(axis=0) 
    class_1_maps = attn_maps[all_targets == 1].mean(axis=0) 
    
    # 5. Save Results
    os.makedirs(output_dir, exist_ok=True)
    
    np.savetxt(os.path.join(output_dir, "class_0_maps.csv"), class_0_maps, delimiter=",")
    np.savetxt(os.path.join(output_dir, "class_1_maps.csv"), class_1_maps, delimiter=",")
    
    with open(os.path.join(output_dir, "ch_names.txt"), "w") as f:
        for ch in ch_names:
            f.write(f"{ch}\n")

    print(f"Extraction complete. Data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Attention Weights from LaBraM Checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the directory containing checkpoint-best.pth")
    parser.add_argument("--output_dir", type=str, default="./attention_data", help="Directory to save the CSV/TXT files")
    args = parser.parse_args()

    extract_and_save_attention(args.checkpoint_dir, args.output_dir)
    