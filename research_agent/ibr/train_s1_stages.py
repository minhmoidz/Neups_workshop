"""Phase-II IBR S1 - locked two-stage training driver (STEP 6C.1 amendment).

Implements EXACTLY the STEP 6C.1 locked schedule:

  Stage A - self-reconstruction pretrain (20 epochs, L_rec ONLY on x_self)
  Stage B - full S1 training (30 epochs, full 5-term S1 objective)
             initialised from the Stage-A epoch-20 checkpoint.

Frozen configuration (STEP 6A lock #6 / #11 / #12):
  lambdas            : all 1.0, GRL lambda 1.0
  optimizers         : Adam lr=1e-4 for E/G/V ; Adam lr=1e-4 for H_med
  batch size         : 16
  z_id dimension     : 128
  donor protocol     : deterministic DonorSampler(seed=42), TRAIN/validation pools
  frozen utility     : DenseNet-121 classifier + UNetSeg teacher (SHA hard-fail)

Checkpoint-selection rule (STEP 6C.1):
  - NO validation-based selection.
  - Frozen checkpoint = LAST NUMERICALLY VALID COMPLETED checkpoint of
    Stage B epoch 30.
  - All per-epoch validation numbers are DIAGNOSTICS ONLY.

Usage:
    PYTHONPATH=. .venv/bin/python research_agent/ibr/train_s1_stages.py \
        --seed 42 --bs 16 --out_dir research_agent/ibr_s1_oneseed_artifacts \
        --device cuda
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from chexnet.cxr_dataset import CXRDataset
from research_agent.ibr.ibr_model import IBRModel, count_parameters
from research_agent.ibr.frozen_models import (CLASSIFIER_SHA256, SEGMENTATION_SHA256_PREFIX,
                                              load_frozen_classifier,
                                              load_frozen_segmentation_teacher)
from research_agent.ibr.losses import (LAMBDA_REC, LAMBDA_PATH, LAMBDA_ANAT,
                                       LAMBDA_ZID, LAMBDA_ADV, GRL_LAMBDA)
from research_agent.ibr.s1_loss import compute_s1_loss
from research_agent.ibr.donor import DonorSampler

IMAGE_PATH = '/home/minhtt/datasets/nih/images/'
LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
          'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return 'unknown'


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class SingleImageLabels(Dataset):
    """Wraps CXRDataset to return (x in [-1,1], y_path float, image name).

    CXRDataset (perturbation_type='flow_field') yields (1,256,256) in [0,1];
    the S1 model lives in [-1,1] (decoder tanh), so we map x <- 2*x - 1.
    The deterministic donor image (seed 42) is loaded here in the worker
    process so image + donor I/O is parallelised by the DataLoader.
    """

    def __init__(self, fold, seed=42):
        self.ds = CXRDataset(path_to_images=IMAGE_PATH, fold=fold,
                             perturbation_type='flow_field')
        self.sampler = DonorSampler(seed=seed)
        self._donor_cache = {}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label, name = self.ds[idx]
        x = img * 2.0 - 1.0
        y = torch.as_tensor(label, dtype=torch.float32)
        donor = self._donor_cache.get(name)
        if donor is None:
            donor = self.sampler.donor_for(name)
            self._donor_cache[name] = donor
        x_donor = load_x_image(os.path.join(IMAGE_PATH, donor))
        return x, y, name, x_donor, donor


def load_x_image(path):
    from PIL import Image
    from torchvision import transforms
    t = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    im = Image.open(path).convert('L')
    return t(im) * 2.0 - 1.0


class S1Trainer:
    def __init__(self, seed=42, bs=16, device='cuda', out_dir=None):
        self.seed = int(seed)
        self.bs = int(bs)
        self.device = device
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.sampler = DonorSampler(seed=self.seed)
        self.model = IBRModel().to(device)
        self.frozen = _load_frozen(device)

        # Locked optimizers (STEP 6A #11): E/G/V lr=1e-4 ; H_med lr=1e-4.
        egv_params = (list(self.model.encoder.parameters())
                      + list(self.model.decoder.parameters())
                      + list(self.model.verifier.parameters()))
        self.opt_egv = torch.optim.Adam(egv_params, lr=1e-4)
        self.opt_adv = torch.optim.Adam(self.model.adv.parameters(), lr=1e-4)

    # ------------------------------------------------------------------ data
    def _loader(self, fold, shuffle):
        ds = SingleImageLabels(fold)
        return DataLoader(ds, batch_size=self.bs, shuffle=shuffle,
                          num_workers=8, pin_memory=True, drop_last=False)

    def _val_subset(self, fold='val', n=256, seed=7):
        """Fixed deterministic validation subset for per-epoch diagnostics."""
        ds = SingleImageLabels(fold)
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[:min(n, len(ds))].tolist()
        return [ds[i] for i in idx]

    # ----------------------------------------------------------------- stage A
    def stage_a(self, n_epochs=20):
        print('=== STAGE A: self-reconstruction pretrain (L_rec only), %d epochs ===' % n_epochs)
        loader = self._loader('train', shuffle=True)
        logs = []
        for ep in range(1, n_epochs + 1):
            self.model.train()
            t0 = time.time()
            run_loss = 0.0
            n_batches = 0
            nan_inf = False
            for x, y_path, names, _, _ in loader:
                x = x.to(self.device)
                self.opt_egv.zero_grad()
                z_id, z_med, skips = self.model.encode(x)
                x_self = self.model.decode(z_id, z_med, skips)
                loss = F.l1_loss(x_self, x)
                loss.backward()
                self.opt_egv.step()
                run_loss += loss.item()
                n_batches += 1
                if not torch.isfinite(loss):
                    nan_inf = True
            train_loss = run_loss / max(n_batches, 1)
            peak = torch.cuda.max_memory_allocated() / 1e9
            diag = self._validation_diagnostics(prefix='stageA')
            rec = {'stage': 'A', 'epoch': ep, 'train_loss_rec': train_loss,
                   'nan_inf': nan_inf, 'lr': 1e-4, 'peak_gpu_gb': round(peak, 3),
                   'secs': round(time.time() - t0, 1), 'commit': git_head(),
                   'seed': self.seed, 'bs': self.bs, 'timestamp': utcnow()}
            rec.update(diag)
            logs.append(rec)
            print('  A ep %02d  L_rec %.4f  %s' % (ep, train_loss,
                                                   {k: round(v, 4) for k, v in diag.items()
                                                    if isinstance(v, float)}))
            self._save_logs('stage_a_logs.json', logs)
        ckpt = self._make_checkpoint(stage='A', epoch=n_epochs)
        self._save_checkpoint('stage_a_epoch%02d.pth' % n_epochs, ckpt)
        print('  STAGE A DONE. checkpoint=%s' % ckpt['path'])
        return ckpt

    # ----------------------------------------------------------------- stage B
    def stage_b(self, n_epochs=30, init_checkpoint=None):
        print('=== STAGE B: full S1 training, %d epochs ===' % n_epochs)
        if init_checkpoint is not None:
            self._load_checkpoint(init_checkpoint)
        loader = self._loader('train', shuffle=True)
        logs = []
        for ep in range(1, n_epochs + 1):
            self.model.train()
            t0 = time.time()
            acc = {'L_rec': 0.0, 'L_path': 0.0, 'L_anat': 0.0, 'L_zid': 0.0, 'L_adv': 0.0, 'total': 0.0}
            n_batches = 0
            nan_inf = False
            donor_same = 0
            for x, y_path, names, x_donor, donor_names in loader:
                x = x.to(self.device)
                x_donor = x_donor.to(self.device)
                y_path = y_path.to(self.device)
                y_pair = torch.zeros(x.shape[0], 1, device=self.device)
                self.opt_egv.zero_grad()
                self.opt_adv.zero_grad()
                total, parts = compute_s1_loss(self.model, self.frozen, x, x_donor,
                                               y_path, y_pair, return_parts=True)
                total.backward()
                self.opt_egv.step()
                self.opt_adv.step()
                for k in acc:
                    acc[k] += parts[k]
                n_batches += 1
                if not torch.isfinite(total):
                    nan_inf = True
                prov = self.sampler.provenance(names, donor_names)
                donor_same += sum(1 for p in prov if p['donor_patient'] == p['source_patient'])
            train = {k: v / max(n_batches, 1) for k, v in acc.items()}
            peak = torch.cuda.max_memory_allocated() / 1e9
            diag = self._validation_diagnostics(prefix='stageB')
            rec = {'stage': 'B', 'epoch': ep, 'train_total_loss': train['total'],
                   'train_L_rec': train['L_rec'], 'train_L_path': train['L_path'],
                   'train_L_anat': train['L_anat'], 'train_L_zid': train['L_zid'],
                   'train_L_adv': train['L_adv'],
                   'donor_equals_source_count': donor_same,
                   'nan_inf': nan_inf, 'lr': 1e-4, 'peak_gpu_gb': round(peak, 3),
                   'secs': round(time.time() - t0, 1), 'commit': git_head(),
                   'seed': self.seed, 'bs': self.bs, 'timestamp': utcnow()}
            rec.update(diag)
            logs.append(rec)
            print('  B ep %02d  total %.4f [rec %.4f path %.4f anat %.4f zid %.4f adv %.4f] %s'
                  % (ep, train['total'], train['L_rec'], train['L_path'], train['L_anat'],
                     train['L_zid'], train['L_adv'],
                     {k: round(v, 4) for k, v in diag.items() if isinstance(v, float)}))
            self._save_logs('stage_b_logs.json', logs)
            self._save_checkpoint('stage_b_epoch%02d.pth' % ep, self._make_checkpoint(stage='B', epoch=ep))
        ckpt = self._make_checkpoint(stage='B', epoch=n_epochs)
        self._save_checkpoint('stage_b_epoch%02d_FROZEN.pth' % n_epochs, ckpt)
        print('  STAGE B DONE. FROZEN checkpoint=%s' % ckpt['path'])
        return ckpt

    # ------------------------------------------------------------- diagnostics
    def _validation_diagnostics(self, prefix='stageA', n=256):
        """Per-epoch validation diagnostics (DIAGNOSTICS ONLY; never selects).

        Computed on a fixed deterministic validation subset for tractability.
        """
        out = {}
        try:
            self.model.eval()
            val = self._val_subset('val', n=n)
            xs = torch.stack([v[0] for v in val]).to(self.device)
            ys = torch.stack([v[1] for v in val]).to(self.device)
            x_donors = torch.stack([v[3] for v in val]).to(self.device)
            with torch.no_grad():
                z_id, z_med, skips = self.model.encode(xs)
                x_self = self.model.decode(z_id, z_med, skips)
                out[prefix + '_val_recon_l1'] = round(float(F.l1_loss(x_self, xs).item()), 5)
                # full S1 loss diagnostics (preloaded deterministic donors)
                y_pair = torch.zeros(xs.shape[0], 1, device=self.device)
                total, parts = compute_s1_loss(self.model, self.frozen, xs, x_donors,
                                               ys, y_pair, return_parts=True)
                out[prefix + '_val_total'] = round(float(total.item()), 5)
                for k in ['L_rec', 'L_path', 'L_anat', 'L_zid', 'L_adv']:
                    out[prefix + '_val_' + k] = round(float(parts[k]), 5)
                # frozen classification diagnostic (x_anon vs source labels)
                out[prefix + '_val_class_auc14'] = self._class_auc14(x_self)  # recon proxy
                # z_id verifier on validation pairs
                vz = self._pair_auc(model_side='zid')
                out[prefix + '_zid_verifier_auc'] = round(vz, 5)
                # z_med adversary on validation pairs
                vm = self._pair_auc(model_side='zmed')
                out[prefix + '_zmed_adversary_auc'] = round(vm, 5)
        except Exception as e:  # diagnostics must never kill training
            out[prefix + '_diagnostic_error'] = str(e)[:200]
        return out

    def _class_auc14(self, x_self):
        """Frozen classifier mean AUC-14 on x_self vs source labels (subset proxy)."""
        try:
            with torch.no_grad():
                probs = self.frozen.path_logits(x_self).detach().cpu().numpy()
            val = self._val_subset('val', n=x_self.shape[0], seed=9)
            ys = torch.stack([v[1] for v in val]).numpy()
            aucs = []
            for i in range(14):
                if len(np.unique(ys[:, i])) > 1:
                    try:
                        aucs.append(roc_auc_score(ys[:, i], probs[:, i]))
                    except ValueError:
                        pass
            return round(float(np.mean(aucs)), 5) if aucs else float('nan')
        except Exception:
            return float('nan')

    def _pair_auc(self, model_side='zid', n=200):
        """AUC of the z_id verifier / z_med adversary on validation image pairs."""
        import numpy as np
        try:
            pairs = np.loadtxt('image_pairs/image_pairs_validation_2000.txt', dtype=str)
            same = [r for r in pairs if r[2] == '1.0'][:n // 2]
            diff = [r for r in pairs if r[2] == '0.0'][:n // 2]
            sel = same + diff
            y, scores = [], []
            with torch.no_grad():
                for row in sel:
                    im1 = load_x_image(os.path.join(IMAGE_PATH, row[0])).unsqueeze(0).to(self.device)
                    im2 = load_x_image(os.path.join(IMAGE_PATH, row[1])).unsqueeze(0).to(self.device)
                    z1, m1, _ = self.model.encode(im1)
                    z2, m2, _ = self.model.encode(im2)
                    if model_side == 'zid':
                        s = torch.sigmoid(self.model.verify(z1, z2)).item()
                    else:
                        s = torch.sigmoid(self.model.adversary_logits(m1, m2)).item()
                    y.append(float(row[2]))
                    scores.append(s)
            if len(set(y)) < 2:
                return float('nan')
            return float(roc_auc_score(y, scores))
        except Exception as e:
            return float('nan')

    # ------------------------------------------------------------- checkpoint
    def _make_checkpoint(self, stage, epoch):
        path = os.path.join(self.out_dir, 's1_%s_epoch%02d.pth' % (stage.lower(), epoch))
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        torch.save({'model': state, 'stage': stage, 'epoch': epoch, 'seed': self.seed,
                    'bs': self.bs, 'lambdas': {'rec': LAMBDA_REC, 'path': LAMBDA_PATH,
                                               'anat': LAMBDA_ANAT, 'zid': LAMBDA_ZID,
                                               'adv': LAMBDA_ADV, 'grl': GRL_LAMBDA},
                    'commit': git_head(), 'timestamp': utcnow(),
                    'classifier_sha256': CLASSIFIER_SHA256,
                    'seg_prefix': SEGMENTATION_SHA256_PREFIX}, path)
        return {'path': path, 'stage': stage, 'epoch': epoch, 'sha256': sha256(path),
                'size_bytes': os.path.getsize(path)}

    def _save_checkpoint(self, fname, ckpt):
        with open(os.path.join(self.out_dir, fname + '.sha256.json'), 'w') as f:
            json.dump(ckpt, f, indent=2)
        print('    saved %s sha=%s size=%d' % (ckpt['path'], ckpt['sha256'][:16], ckpt['size_bytes']))

    def _load_checkpoint(self, ckpt):
        state = torch.load(ckpt['path'], weights_only=False, map_location=self.device)['model']
        self.model.load_state_dict(state, strict=True)
        print('    loaded Stage-%s epoch-%d checkpoint' % (ckpt['stage'], ckpt['epoch']))

    def _save_logs(self, fname, logs):
        with open(os.path.join(self.out_dir, fname), 'w') as f:
            json.dump(logs, f, indent=2)


def _load_frozen(device):
    from research_agent.ibr.losses import FrozenUtility
    return FrozenUtility(device)


def main():
    ap = argparse.ArgumentParser(description='IBR S1 locked two-stage training (STEP 6C.1).')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--epochs_a', type=int, default=20)
    ap.add_argument('--epochs_b', type=int, default=30)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True  # fixed 256x256 input; benchmark autotune

    trainer = S1Trainer(seed=args.seed, bs=args.bs, device=args.device, out_dir=args.out_dir)
    meta = {'seed': args.seed, 'bs': args.bs, 'epochs_a': args.epochs_a, 'epochs_b': args.epochs_b,
            'commit': git_head(), 'device': args.device,
            's1_trainable_params': count_parameters(trainer.model),
            'classifier_sha256': CLASSIFIER_SHA256, 'seg_prefix': SEGMENTATION_SHA256_PREFIX,
            'started': utcnow()}
    with open(os.path.join(args.out_dir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('META', json.dumps(meta))

    a = trainer.stage_a(n_epochs=args.epochs_a)
    b = trainer.stage_b(n_epochs=args.epochs_b, init_checkpoint=a)
    summary = {'meta': meta, 'stage_a_ckpt': a, 'stage_b_frozen_ckpt': b, 'finished': utcnow()}
    with open(os.path.join(args.out_dir, 'run_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('RUN COMPLETE. FROZEN S1 CHECKPOINT:', b['path'])


if __name__ == '__main__':
    main()
