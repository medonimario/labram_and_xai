import os
import glob
import pickle
import torch
import numpy as np
from einops import rearrange
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../labram_ft")))

import utils
from run_class_finetuning import get_models

load_dotenv()

class MockArgs:
    """
    Simulates the argparse object required by `get_models` so we can initialize 
    the base architecture without a finetuned checkpoint dict.
    Matches your bash script parameters.
    """
    def __init__(self):
        self.model = 'labram_base_patch200_200'
        self.nb_classes = 0  # 0 for base feature extraction (no classification head)
        self.drop = 0.0
        self.drop_path = 0.2
        self.attn_drop_rate = 0.0
        self.use_mean_pooling = True
        self.use_attention_pooling = False
        self.init_scale = 0.001
        self.rel_pos_bias = False  # --disable_rel_pos_bias
        self.abs_pos_emb = True    # --abs_pos_emb
        self.layer_scale_init_value = 0.1
        self.qkv_bias = False      # --disable_qkv_bias


class BulkActivationExtractor:
    def __init__(self, pretrained_weights_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Initializing LaBraM-Base on {self.device}...")

        # 1. Initialize model using MockArgs
        self.args = MockArgs()
        self.model = get_models(self.args)
        
        # 2. Load Base Pretrained Weights
        checkpoint = torch.load(pretrained_weights_path, map_location='cpu')
        # Handle different save formats (model, module, or bare dict)
        state_dict = checkpoint.get('model', checkpoint.get('module', checkpoint))
        
        # Remove head weights if they exist in the checkpoint but not in our nb_classes=0 model
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
        
        utils.load_state_dict(self.model, state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("Pretrained base weights loaded successfully.")

        # 3. Setup Channels (Must match 'standard_channels' from your dataset creation script)
        self.ch_names = [
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
        # Capitalize for the utils lookup just in case
        self.ch_names_upper = [ch.upper() for ch in self.ch_names]
        self.input_chans = utils.get_input_chans(self.ch_names_upper)

    @torch.no_grad()
    def get_layer11_embedding(self, eeg_tensor):
        """
        Processes a single (64, 800) window and returns the Layer 11 embedding.
        """
        if eeg_tensor.ndim == 2:
            eeg_tensor = eeg_tensor.unsqueeze(0) 
            
        # Reshape and scale (matching training pipeline)
        eeg_tensor = rearrange(eeg_tensor, 'B N (A T) -> B N A T', T=200)
        eeg_tensor = eeg_tensor.float().to(self.device) / 100

        # Extract intermediate layer 11 (the final transformer block before the head)
        layer_outputs = self.model.forward_intermediate(
            eeg_tensor, 
            layer_id=[11], 
            norm_output=True, 
            input_chans=self.input_chans
        )
        
        # Return as 1D numpy array
        return layer_outputs[0].squeeze().cpu().numpy()


def main():
    # --- Configuration ---
    BASE_WEIGHTS_PATH = './src/labram_ft/checkpoints/labram-base.pth'
    DATASET_ROOT = os.path.join(os.getenv("ENGAGEMENT_DATASET_PATH", ""), "processed")
    OUTPUT_ROOT = os.path.join(os.getenv("ENGAGEMENT_DATASET_PATH", ""), "embeddings_base")
    
    extractor = BulkActivationExtractor(BASE_WEIGHTS_PATH)
    
    # We will process train, and test splits (val is the same as train for this dataset)
    splits = ['train', 'test']
    
    for split in splits:
        split_dir = os.path.join(DATASET_ROOT, split)
        out_split_dir = os.path.join(OUTPUT_ROOT, split)
        os.makedirs(out_split_dir, exist_ok=True)
        
        # Find all .pkl files in this split
        pkl_files = glob.glob(os.path.join(split_dir, "*.pkl"))
        print(f"\nProcessing {len(pkl_files)} files in '{split}' split...")
        
        for file_path in tqdm(pkl_files):
            # Load the segmented EEG data
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            eeg_data = torch.from_numpy(data['X']) # Shape: (64, 800)
            
            # Extract embedding
            embedding = extractor.get_layer11_embedding(eeg_data)
            
            # Save the embedding as a .npy file with the exact same base name
            base_name = os.path.basename(file_path).replace('.pkl', '.npy')
            save_path = os.path.join(out_split_dir, base_name)
            
            np.save(save_path, embedding)

    print("\nEmbedding extraction complete! All vectors saved to:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()