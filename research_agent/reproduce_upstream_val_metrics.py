"""STEP 8R — Upstream Train/Validation Reproduction Script

Executes the original upstream evaluation and retraining pipelines on:
- 10,000 TRAIN pairs
- 2,000 VALIDATION pairs
STRICTLY NO TEST EVALUATION.

Evaluates:
1. Clean Validation Re-ID AUC (Frozen Pretrained SNN)
2. Anonymized Validation Re-ID AUC (Frozen Pretrained SNN + Upstream Generator mu=0.01)
3. Retrained Attacker Validation Re-ID AUC (SNN retrained on anonymized Train pairs, evaluated on Val pairs)
4. Classification Validation Macro AUC & 14-disease AUCs (Clean vs Anonymized)
5. Comparison against Corrected Operator on Validation set.
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
    def __init__(self, pairs_file, n_channels=1, image_size=256, max_pairs=None):
        self.image_size = image_size
        self.n_channels = n_channels
        self.resize = transforms.Resize((image_size, image_size))
        self.transform = transforms.ToTensor() if n_channels == 1 else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        raw_pairs = np.loadtxt(pairs_file, dtype=str)
        if max_pairs is not None:
            raw_pairs = raw_pairs[:max_pairs]
        self.pairs = raw_pairs
        
        self.imgs1 = []
        self.imgs2 = []
        self.labels = []
        
        print(f"Loading {len(self.pairs)} pairs from {pairs_file}...")
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
        print(f"Loaded in {time.time() - t0:.2f}s")

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


def eval_reid_on_loader(snn, val_loader, generator=None, grid_identity=None, gauss_filter=None, mu=0.01, mode='legacy', device='cuda'):
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
                
            # Expand to 3 channels and normalize ImageNet
            im1_3c = normalize(im1.expand(-1, 3, -1, -1))
            im2_3c = normalize(im2.expand(-1, 3, -1, -1))
            
            out = snn(im1_3c, im2_3c).squeeze()
            probs = torch.sigmoid(out).cpu().numpy().flatten()
            
            y_true_list.extend(lbl.numpy().flatten().tolist())
            y_pred_list.extend(probs.tolist())
            
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    auc = float(metrics.roc_auc_score(y_true, y_pred))
    return auc, y_true, y_pred


def train_and_eval_retrained_attacker(train_loader, val_loader, generator, grid_identity, gauss_filter, mu=0.01, mode='legacy', max_epochs=20, patience=5, device='cuda'):
    snn = SiameseNetwork().to(device)
    optimizer = optim.Adam(snn.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss().to(device)
    es = EarlyStopping(patience=patience)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    best_loss = 1e9
    best_snn_state = copy.deepcopy(snn.state_dict())
    best_epoch = 0
    
    generator.eval()
    print(f"\n--- Retraining Attacker SNN on Train Pairs (mode={mode}, max_epochs={max_epochs}) ---")
    
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
        
        # Validation loss
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
        
        print(f"Epoch [{epoch+1:02d}/{max_epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_snn_state = copy.deepcopy(snn.state_dict())
            best_epoch = epoch + 1
            
        if es.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    # Load best network
    snn.load_state_dict(best_snn_state)
    final_val_auc, _, _ = eval_reid_on_loader(snn, val_loader, generator, grid_identity, gauss_filter, mu=mu, mode=mode, device=device)
    print(f"Best Attacker (Epoch {best_epoch}, Best Val Loss {best_loss:.4f}): Final Val AUC = {final_val_auc:.4f}")
    
    return {
        'best_epoch': best_epoch,
        'best_val_loss': float(best_loss),
        'final_val_auc': float(final_val_auc),
        'checkpoint_selection_criterion': 'lowest_validation_bce_loss'
    }


def eval_classification_utility_on_val(ac_model, generator, grid_identity, gauss_filter, val_pairs_file, mu=0.01, device='cuda'):
    print("\n--- Evaluating Classification Utility on Validation Set ---")
    ac_model.eval()
    generator.eval()
    
    NIH_LABELS = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
        'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    
    meta_df = pd.read_csv("/home/minhtt/PriCheXy-Net_upstream_reproduction/Data_Entry_2017_v2020.csv")
    meta_dict = dict(zip(meta_df['Image Index'], meta_df['Finding Labels']))
    
    pairs = np.loadtxt(val_pairs_file, dtype=str)
    # Collect unique images from val pairs
    val_images = sorted(list(set(pairs[:, 0]).union(set(pairs[:, 1]))))
    print(f"Unique validation images to classify: {len(val_images)}")
    
    resize_256 = transforms.Resize((256, 256))
    resize_224 = transforms.Resize((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    y_true = []
    y_pred_clean = []
    y_pred_anon_legacy = []
    y_pred_anon_corr = []
    
    batch_size = 32
    for start_idx in range(0, len(val_images), batch_size):
        batch_imgs = val_images[start_idx:start_idx + batch_size]
        
        tensors = []
        lbls = []
        for img_name in batch_imgs:
            p = os.path.join(IMAGE_PATH, img_name)
            pil_img = resize_256(pil_loader(p, 1))
            tensors.append(to_tensor(pil_img))
            
            # Label
            finding = meta_dict.get(img_name, 'No Finding')
            lbl_vec = np.zeros(14, dtype=np.float32)
            if finding != 'No Finding':
                for f in finding.split('|'):
                    if f in NIH_LABELS:
                        lbl_vec[NIH_LABELS.index(f)] = 1.0
            lbls.append(lbl_vec)
            
        inp = torch.stack(tensors, dim=0).to(device)
        lbls = np.stack(lbls, axis=0)
        y_true.append(lbls)
        
        with torch.no_grad():
            # Clean
            clean_3c = normalize(resize_224(inp.expand(-1, 3, -1, -1)))
            out_clean = torch.sigmoid(ac_model(clean_3c)).cpu().numpy()
            y_pred_clean.append(out_clean)
            
            # Anonymized Legacy
            anon_leg = deform_images(inp, generator, grid_identity, gauss_filter, mu=mu, mode='legacy')
            anon_leg_3c = normalize(resize_224(anon_leg.expand(-1, 3, -1, -1)))
            out_leg = torch.sigmoid(ac_model(anon_leg_3c)).cpu().numpy()
            y_pred_anon_legacy.append(out_leg)
            
            # Anonymized Corrected
            anon_cor = deform_images(inp, generator, grid_identity, gauss_filter, mu=mu, mode='corrected')
            anon_cor_3c = normalize(resize_224(anon_cor.expand(-1, 3, -1, -1)))
            out_cor = torch.sigmoid(ac_model(anon_cor_3c)).cpu().numpy()
            y_pred_anon_corr.append(out_cor)
            
    y_true = np.concatenate(y_true, axis=0)
    y_pred_clean = np.concatenate(y_pred_clean, axis=0)
    y_pred_anon_legacy = np.concatenate(y_pred_anon_legacy, axis=0)
    y_pred_anon_corr = np.concatenate(y_pred_anon_corr, axis=0)
    
    auc_clean = {}
    auc_legacy = {}
    auc_corr = {}
    
    for idx, lbl in enumerate(NIH_LABELS):
        # Only compute if at least 1 positive and 1 negative
        if np.sum(y_true[:, idx]) > 0 and np.sum(1 - y_true[:, idx]) > 0:
            auc_clean[lbl] = float(metrics.roc_auc_score(y_true[:, idx], y_pred_clean[:, idx]))
            auc_legacy[lbl] = float(metrics.roc_auc_score(y_true[:, idx], y_pred_anon_legacy[:, idx]))
            auc_corr[lbl] = float(metrics.roc_auc_score(y_true[:, idx], y_pred_anon_corr[:, idx]))
            
    macro_clean = float(np.mean(list(auc_clean.values())))
    macro_legacy = float(np.mean(list(auc_legacy.values())))
    macro_corr = float(np.mean(list(auc_corr.values())))
    
    print(f"Validation Macro AUC: Clean = {macro_clean:.4f} | Anonymized Legacy = {macro_legacy:.4f} (Delta = {macro_legacy - macro_clean:.4f}) | Corrected = {macro_corr:.4f} (Delta = {macro_corr - macro_clean:.4f})")
    
    return {
        'macro_auc_clean': macro_clean,
        'macro_auc_anonymized_legacy': macro_legacy,
        'macro_auc_anonymized_corrected': macro_corr,
        'delta_macro_auc_legacy': macro_legacy - macro_clean,
        'delta_macro_auc_corrected': macro_corr - macro_clean,
        'per_label_clean': auc_clean,
        'per_label_legacy': auc_legacy,
        'per_label_corrected': auc_corr,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load upstream models
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
    clean_val_reid_auc, _, _ = eval_reid_on_loader(snn_pretrained, val_loader, generator=None, device=device)
    print(f"\n[1] Clean Validation Re-ID AUC (Frozen Pretrained SNN): {clean_val_reid_auc:.4f}")
    
    # 4. Anonymized Validation Re-ID (Frozen Pretrained SNN + Legacy Operator)
    anon_legacy_val_reid_frozen_auc, _, _ = eval_reid_on_loader(snn_pretrained, val_loader, generator=gen, grid_identity=grid_identity, gauss_filter=gauss_filter, mu=0.01, mode='legacy', device=device)
    print(f"[2] Anonymized Validation Re-ID AUC (Frozen Pretrained SNN + Legacy): {anon_legacy_val_reid_frozen_auc:.4f}")
    
    # 5. Anonymized Validation Re-ID (Frozen Pretrained SNN + Corrected Operator)
    anon_corr_val_reid_frozen_auc, _, _ = eval_reid_on_loader(snn_pretrained, val_loader, generator=gen, grid_identity=grid_identity, gauss_filter=gauss_filter, mu=0.01, mode='corrected', device=device)
    print(f"[3] Anonymized Validation Re-ID AUC (Frozen Pretrained SNN + Corrected): {anon_corr_val_reid_frozen_auc:.4f}")
    
    # 6. Retrained Attacker Validation Re-ID (Legacy Upstream Bundle)
    retrain_legacy_results = train_and_eval_retrained_attacker(train_loader, val_loader, gen, grid_identity, gauss_filter, mu=0.01, mode='legacy', max_epochs=20, patience=5, device=device)
    
    # 7. Retrained Attacker Validation Re-ID (Corrected Operator Bundle)
    retrain_corr_results = train_and_eval_retrained_attacker(train_loader, val_loader, gen, grid_identity, gauss_filter, mu=0.01, mode='corrected', max_epochs=20, patience=5, device=device)
    
    # 8. Classification Utility Evaluation on Validation Set
    class_results = eval_classification_utility_on_val(ac_model, gen, grid_identity, gauss_filter, val_pairs_file, mu=0.01, device=device)
    
    full_results = {
        'clean_validation_reid_auc_frozen_snn': clean_val_reid_auc,
        'anonymized_validation_reid_frozen_snn': {
            'legacy_upstream_operator': anon_legacy_val_reid_frozen_auc,
            'corrected_operator': anon_corr_val_reid_frozen_auc,
        },
        'anonymized_validation_reid_retrained_attacker': {
            'legacy_upstream_operator': retrain_legacy_results,
            'corrected_operator': retrain_corr_results,
        },
        'validation_classification_utility': class_results,
        'summary_table': {
            'Clean Validation Re-ID AUC': clean_val_reid_auc,
            'Anonymized (Frozen Attacker, Legacy Operator) AUC': anon_legacy_val_reid_frozen_auc,
            'Anonymized (Retrained Attacker, Legacy Operator) AUC': retrain_legacy_results['final_val_auc'],
            'Anonymized (Frozen Attacker, Corrected Operator) AUC': anon_corr_val_reid_frozen_auc,
            'Anonymized (Retrained Attacker, Corrected Operator) AUC': retrain_corr_results['final_val_auc'],
            'Clean Classification Macro AUC': class_results['macro_auc_clean'],
            'Anonymized Classification Macro AUC (Legacy)': class_results['macro_auc_anonymized_legacy'],
            'Anonymized Classification Macro AUC (Corrected)': class_results['macro_auc_anonymized_corrected'],
        }
    }
    
    out_path = '/home/minhtt/Neups_workshop/research_agent/validation_reproduction_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(full_results, f, indent=2)
        
    print(f"\nSaved all validation reproduction metrics to {out_path}")
    print(json.dumps(full_results['summary_table'], indent=2))


if __name__ == '__main__':
    main()
