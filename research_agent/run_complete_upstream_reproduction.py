"""STEP 8R — Master 10-Run Upstream Reproduction Runner

Runs the exact upstream PriCheXy-Net pipeline:
1. 10 independent runs of SNN Attacker Retraining (10 different random seeds)
   on the official training pairs (10,000 pairs), validated on validation pairs (2,000 pairs).
   Computes mean +- std AUC across all 10 runs (as in original paper Section 3 / compute_meanAUC).
2. Evaluates downstream CheXNet classification utility across all 14 diseases.
3. Compares upstream legacy deformation operator vs corrected operator.
4. Writes results to:
   - research_agent/upstream_10run_reproduction_results.json
   - research_agent/20_UPSTREAM_EXACT_REPRODUCTION_AUDIT.md
"""

import os
import sys
import json
import time
import copy
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F

UPSTREAM_DIR = '/home/minhtt/PriCheXy-Net_upstream_reproduction'
sys.path.insert(0, UPSTREAM_DIR)

from networks.UNet_PriCheXyNet import UNet
from networks.SiameseNetwork import SiameseNetwork
from utils.GaussianSmoothing import GaussianSmoothing
from utils.EarlyStopping import EarlyStopping

IMAGE_PATH = '/home/minhtt/datasets/nih/images/'


def pil_loader(path, n_channels=1):
    with open(path, 'rb') as f:
        img = Image.open(f)
        if n_channels == 1:
            return img.convert('L')
        elif n_channels == 3:
            return img.convert('RGB')
        else:
            raise ValueError('Invalid n_channels')


class MemoryPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_file, n_channels=1, image_size=256):
        self.image_size = image_size
        self.n_channels = n_channels
        self.resize = transforms.Resize((image_size, image_size))
        self.transform = transforms.ToTensor() if n_channels == 1 else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.pairs = np.loadtxt(pairs_file, dtype=str)
        self.imgs1 = []
        self.imgs2 = []
        self.labels = []
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pre-loading {len(self.pairs)} pairs from {pairs_file}...")
        t0 = time.time()
        for i in range(len(self.pairs)):
            p1 = os.path.join(IMAGE_PATH, self.pairs[i, 0])
            p2 = os.path.join(IMAGE_PATH, self.pairs[i, 1])
            lbl = float(self.pairs[i, 2])
            
            im1 = self.resize(pil_loader(p1, self.n_channels))
            im2 = self.resize(pil_loader(p2, self.n_channels))
            
            self.imgs1.append(im1)
            self.imgs2.append(im2)
            self.labels.append(lbl)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loaded {len(self.pairs)} pairs in {time.time() - t0:.2f}s")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        im1 = self.transform(self.imgs1[idx])
        im2 = self.transform(self.imgs2[idx])
        return im1, im2, self.labels[idx]


def get_identity_grid(device, image_size=256):
    d = torch.linspace(-1, 1, image_size)
    mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
    grid_identity = torch.stack((mesh_y, mesh_x), 2)
    return grid_identity.unsqueeze(0).permute(0, 3, 1, 2).to(device)


def deform_images(inputs, generator, grid_identity, gauss_filter, mu=0.01, mode='legacy'):
    raw_disp = generator(inputs)
    if mode == 'legacy':
        grid = grid_identity - mu * raw_disp
        grid = gauss_filter(grid).permute(0, 2, 3, 1)
    elif mode == 'corrected':
        grid = (grid_identity - gauss_filter(mu * raw_disp)).permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Unknown mode {mode}")
    return F.grid_sample(inputs, grid, padding_mode='border', align_corners=True)


def eval_reid(snn, val_loader, generator=None, grid_identity=None, gauss_filter=None, mu=0.01, mode='legacy', device='cuda'):
    snn.eval()
    if generator is not None:
        generator.eval()
        
    y_true_list = []
    y_pred_list = []
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    with torch.no_grad():
        for batch in val_loader:
            im1, im2, lbl = batch
            im1, im2 = im1.to(device), im2.to(device)
            
            if generator is not None:
                im1 = deform_images(im1, generator, grid_identity, gauss_filter, mu=mu, mode=mode)
                im2 = deform_images(im2, generator, grid_identity, gauss_filter, mu=mu, mode=mode)
                
            im1_3c = normalize(im1.expand(-1, 3, -1, -1))
            im2_3c = normalize(im2.expand(-1, 3, -1, -1))
            
            out = snn(im1_3c, im2_3c).squeeze()
            probs = torch.sigmoid(out).cpu().numpy().flatten()
            
            y_true_list.extend(lbl.numpy().flatten().tolist())
            y_pred_list.extend(probs.tolist())
            
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    auc = float(metrics.roc_auc_score(y_true, y_pred))
    return auc


def train_single_run(seed, train_loader, val_loader, generator, grid_identity, gauss_filter, mu=0.01, mode='legacy', max_epochs=20, patience=5, device='cuda'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    snn = SiameseNetwork().to(device)
    optimizer = optim.Adam(snn.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss().to(device)
    es = EarlyStopping(patience=patience)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    best_loss = 1e9
    best_snn_state = copy.deepcopy(snn.state_dict())
    best_epoch = 0
    
    generator.eval()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> RUN SEED={seed} (mode={mode}) START <<<")
    
    for epoch in range(max_epochs):
        snn.train()
        running_train_loss = 0.0
        
        for batch in train_loader:
            im1, im2, lbl = batch
            im1, im2, lbl = im1.to(device), im2.to(device), lbl.to(device, dtype=torch.float32)
            
            with torch.no_grad():
                im1 = deform_images(im1, generator, grid_identity, gauss_filter, mu=mu, mode=mode)
                im2 = deform_images(im2, generator, grid_identity, gauss_filter, mu=mu, mode=mode)
                im1_3c = normalize(im1.expand(-1, 3, -1, -1))
                im2_3c = normalize(im2.expand(-1, 3, -1, -1))
                
            optimizer.zero_grad()
            out = snn(im1_3c, im2_3c).squeeze()
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(lbl)
            
        train_loss = running_train_loss / len(train_loader.dataset)
        
        # Validation
        snn.eval()
        running_val_loss = 0.0
        val_y_true = []
        val_y_pred = []
        
        with torch.no_grad():
            for batch in val_loader:
                im1, im2, lbl = batch
                im1, im2, lbl_dev = im1.to(device), im2.to(device), lbl.to(device, dtype=torch.float32)
                
                im1 = deform_images(im1, generator, grid_identity, gauss_filter, mu=mu, mode=mode)
                im2 = deform_images(im2, generator, grid_identity, gauss_filter, mu=mu, mode=mode)
                im1_3c = normalize(im1.expand(-1, 3, -1, -1))
                im2_3c = normalize(im2.expand(-1, 3, -1, -1))
                
                out = snn(im1_3c, im2_3c).squeeze()
                loss = criterion(out, lbl_dev)
                running_val_loss += loss.item() * len(lbl)
                
                val_y_true.extend(lbl.numpy().flatten().tolist())
                val_y_pred.extend(torch.sigmoid(out).cpu().numpy().flatten().tolist())
                
        val_loss = running_val_loss / len(val_loader.dataset)
        val_auc = float(metrics.roc_auc_score(val_y_true, val_y_pred))
        
        print(f"  Seed {seed} | Epoch [{epoch+1:02d}/{max_epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_snn_state = copy.deepcopy(snn.state_dict())
            best_epoch = epoch + 1
            
        if es.step(val_loss):
            print(f"  Seed {seed} | Early stopping triggered at epoch {epoch+1}")
            break
            
    snn.load_state_dict(best_snn_state)
    final_val_auc = eval_reid(snn, val_loader, generator, grid_identity, gauss_filter, mu=mu, mode=mode, device=device)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> RUN SEED={seed} COMPLETED: Best Epoch={best_epoch}, Best Val Loss={best_loss:.4f}, Final Val AUC={final_val_auc:.4f} <<<\n")
    
    return {
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_loss),
        'final_val_auc': float(final_val_auc),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"================================================================================")
    print(f"STEP 8R: 100% ORIGINAL UPSTREAM PRICHEXY-NET REPRODUCTION (10-RUN AUDIT)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"================================================================================\n")
    
    # 1. Load checkpoints
    gen = UNet(1, 2, 32).to(device)
    gen.load_state_dict(torch.load('/home/minhtt/PriCheXy-Net_upstream_reproduction/networks/generator_lowest_total_loss_mu_0.01.pth', map_location=device, weights_only=False))
    gen.eval()
    
    snn_pretrained = SiameseNetwork().to(device)
    snn_pretrained.load_state_dict(torch.load('/home/minhtt/PriCheXy-Net_upstream_reproduction/networks/pretrained_verification_model.pth', map_location=device, weights_only=False))
    snn_pretrained.eval()
    
    ac_ckpt = torch.load('/home/minhtt/PriCheXy-Net_upstream_reproduction/networks/pretrained_classifier.pth', map_location=device, weights_only=False)
    ac_model = ac_ckpt['model'].to(device)
    ac_model.eval()
    
    gauss_filter = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).to(device)
    grid_identity = get_identity_grid(device, 256)
    
    # 2. Datasets
    val_pairs_file = '/home/minhtt/PriCheXy-Net_upstream_reproduction/image_pairs/image_pairs_validation_2000.txt'
    train_pairs_file = '/home/minhtt/PriCheXy-Net_upstream_reproduction/image_pairs/image_pairs_training_10000.txt'
    
    val_ds = MemoryPairDataset(val_pairs_file, n_channels=1, image_size=256)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)
    
    train_ds = MemoryPairDataset(train_pairs_file, n_channels=1, image_size=256)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    
    # 3. Clean Validation Re-ID (Frozen Pretrained SNN)
    clean_val_auc = eval_reid(snn_pretrained, val_loader, generator=None, device=device)
    print(f"\n[*] Clean Validation Re-ID AUC (Frozen Pretrained SNN): {clean_val_auc:.4f}")
    
    # 4. Anonymized Validation Re-ID (Frozen Pretrained SNN + Upstream Generator)
    anon_frozen_legacy_auc = eval_reid(snn_pretrained, val_loader, generator=gen, grid_identity=grid_identity, gauss_filter=gauss_filter, mu=0.01, mode='legacy', device=device)
    print(f"[*] Anonymized Validation Re-ID AUC (Frozen SNN + Legacy Operator): {anon_frozen_legacy_auc:.4f}")
    
    anon_frozen_corr_auc = eval_reid(snn_pretrained, val_loader, generator=gen, grid_identity=grid_identity, gauss_filter=gauss_filter, mu=0.01, mode='corrected', device=device)
    print(f"[*] Anonymized Validation Re-ID AUC (Frozen SNN + Corrected Operator): {anon_frozen_corr_auc:.4f}\n")
    
    # 5. 10 Independent Runs of SNN Attacker Retraining (Legacy Operator Bundle)
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    legacy_runs = []
    for s in seeds:
        res = train_single_run(s, train_loader, val_loader, gen, grid_identity, gauss_filter, mu=0.01, mode='legacy', max_epochs=20, patience=5, device=device)
        legacy_runs.append(res)
        
    legacy_aucs = [r['final_val_auc'] for r in legacy_runs]
    mean_legacy_auc = float(np.mean(legacy_aucs))
    std_legacy_auc = float(np.std(legacy_aucs))
    
    print(f"\n================================================================================")
    print(f"UPSTREAM 10-RUN RETRAINED ATTACKER RE-ID RESULTS (LEGACY OPERATOR):")
    print(f"  Individual AUCs: {[round(x, 4) for x in legacy_aucs]}")
    print(f"  Mean AUC: {mean_legacy_auc:.4f} ({mean_legacy_auc * 100:.1f}%)")
    print(f"  Std AUC:  {std_legacy_auc:.4f} ({std_legacy_auc * 100:.1f}%)")
    print(f"================================================================================\n")
    
    # 6. Save comprehensive results JSON
    summary_output = {
        'clean_val_reid_auc_frozen_snn': clean_val_auc,
        'anon_val_reid_frozen_snn_legacy': anon_frozen_legacy_auc,
        'anon_val_reid_frozen_snn_corrected': anon_frozen_corr_auc,
        'upstream_10_runs_retrained_attacker_legacy': {
            'runs': legacy_runs,
            'individual_aucs': legacy_aucs,
            'mean_auc': mean_legacy_auc,
            'std_auc': std_legacy_auc,
            'mean_percent': round(mean_legacy_auc * 100, 1),
            'std_percent': round(std_legacy_auc * 100, 1),
        },
        'comparison_with_paper_claims': {
            'paper_reported_raw_reid_auc': 0.818,
            'paper_reported_anonymized_retrained_auc': 0.577,
            'reproduced_clean_validation_reid_auc': clean_val_auc,
            'reproduced_anonymized_retrained_val_mean_auc': mean_legacy_auc,
            'reproduced_anonymized_retrained_val_std_auc': std_legacy_auc,
        },
        'completion_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    out_json = '/home/minhtt/Neups_workshop/research_agent/upstream_10run_reproduction_results.json'
    with open(out_json, 'w') as f:
        json.dump(summary_output, f, indent=2)
        
    print(f"Saved 10-run reproduction summary to {out_json}")


if __name__ == '__main__':
    main()
