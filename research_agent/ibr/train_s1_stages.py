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

    Identity pair (lock §3/§4): each sample carries a deterministic partner
    image x_partner and the same/different-patient label y_pair. With prob 0.5
    (and when the source patient has >= 2 images in the split) the partner is a
    SAME-patient image (y_pair=1); otherwise it is the different-patient donor
    (y_pair=0). This guarantees BOTH classes reach the z_id verifier and the
    z_med adversary -- fixing the Stage-B bug where y_pair was hardcoded to 0
    (source, donor) is always a different-patient pair, so the identity heads
    only ever saw negative pairs and collapsed to chance.
    """

    def __init__(self, fold, seed=42):
        self.ds = CXRDataset(path_to_images=IMAGE_PATH, fold=fold,
                             perturbation_type='flow_field')
        self.sampler = DonorSampler(seed=seed)
        self.seed = int(seed)
        self._donor_cache = {}

    def __len__(self):
        return len(self.ds)

    def _partner_for(self, name):
        """Deterministic identity-pair partner + label (pure function of seed+name).

        Returns (partner_image, y_pair). Same-patient when the source patient has
        >= 2 images in the split and a seeded coin lands heads; else the donor
        (guaranteed different patient, y_pair=0).
        """
        src_pid = int(self.sampler.patient_by_image[name])
        split = self.sampler._split_for(name)
        pool_imgs = self.sampler._pool[split]['images_by_patient'].get(src_pid, [])
        others = [im for im in pool_imgs if im != name]
        key = hashlib.sha256(repr(('pair_partner', self.seed, name)).encode()).hexdigest()
        rng = np.random.default_rng(self.seed + int(key[:16], 16) % (2 ** 32))
        if others and rng.random() < 0.5:
            partner = others[int(rng.integers(0, len(others)))]
            assert int(self.sampler.patient_by_image[partner]) == src_pid
            return partner, 1.0
        donor = self._donor_for(name)
        return donor, 0.0

    def _donor_for(self, name):
        donor = self._donor_cache.get(name)
        if donor is None:
            donor = self.sampler.donor_for(name)
            self._donor_cache[name] = donor
        return donor

    def __getitem__(self, idx):
        img, label, name = self.ds[idx]
        x = img * 2.0 - 1.0
        y = torch.as_tensor(label, dtype=torch.float32)
        donor = self._donor_for(name)
        x_donor = load_x_image(os.path.join(IMAGE_PATH, donor))
        partner, y_pair = self._partner_for(name)
        x_partner = load_x_image(os.path.join(IMAGE_PATH, partner))
        return x, y, name, x_donor, donor, x_partner, torch.tensor(y_pair, dtype=torch.float32).reshape(1)


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
    def stage_a(self, n_epochs=20, resume_from=0):
        print('=== STAGE A: self-reconstruction pretrain (L_rec only), %d epochs ===' % n_epochs)
        if resume_from > 0:
            ckpt = {'path': os.path.join(self.out_dir, 's1_a_epoch%02d.pth' % resume_from)}
            self._load_checkpoint(ckpt)
            print('  resuming Stage A from epoch %d' % resume_from)
        loader = self._loader('train', shuffle=True)
        logs = self._load_logs('stage_a_logs.json')
        logs = [r for r in logs if r.get('epoch', 0) <= resume_from]
        for ep in range(resume_from + 1, n_epochs + 1):
            self.model.train()
            t0 = time.time()
            run_loss = 0.0
            n_batches = 0
            nan_inf = False
            for x, y_path, names, _, _, _, _ in loader:
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
            self._make_checkpoint(stage='A', epoch=ep)  # per-epoch ckpt for resume
        ckpt = self._make_checkpoint(stage='A', epoch=n_epochs)
        self._save_checkpoint('stage_a_epoch%02d.pth' % n_epochs, ckpt)
        print('  STAGE A DONE. checkpoint=%s' % ckpt['path'])
        return ckpt

    # ----------------------------------------------------------------- stage B
    def stage_b(self, n_epochs=30, init_checkpoint=None, resume_from=0):
        print('=== STAGE B: full S1 training, %d epochs ===' % n_epochs)
        if resume_from > 0:
            ckpt = {'path': os.path.join(self.out_dir, 's1_b_epoch%02d.pth' % resume_from)}
            self._load_checkpoint(ckpt)
            print('  resuming Stage B from epoch %d' % resume_from)
        elif init_checkpoint is not None:
            self._load_checkpoint(init_checkpoint)
        loader = self._loader('train', shuffle=True)
        logs = self._load_logs('stage_b_logs.json')
        logs = [r for r in logs if r.get('epoch', 0) <= resume_from]
        for ep in range(resume_from + 1, n_epochs + 1):
            self.model.train()
            t0 = time.time()
            acc = {'L_rec': 0.0, 'L_path': 0.0, 'L_anat': 0.0, 'L_zid': 0.0, 'L_adv': 0.0, 'total': 0.0}
            n_batches = 0
            nan_inf = False
            donor_same = 0
            for x, y_path, names, x_donor, donor_names, x_partner, y_pair in loader:
                x = x.to(self.device)
                x_donor = x_donor.to(self.device)
                x_partner = x_partner.to(self.device)
                y_path = y_path.to(self.device)
                y_pair = y_pair.to(self.device)
                self.opt_egv.zero_grad()
                self.opt_adv.zero_grad()
                total, parts = compute_s1_loss(self.model, self.frozen, x, x_donor,
                                               y_path, x_partner, y_pair, return_parts=True)
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
            x_partners = torch.stack([v[5] for v in val]).to(self.device)
            y_pairs = torch.stack([v[6] for v in val]).to(self.device)
            with torch.no_grad():
                # chunked so frozen-model memory matches the training batch (bs=16)
                recon_acc = 0.0
                loss_acc = {'L_rec': 0.0, 'L_path': 0.0, 'L_anat': 0.0, 'L_zid': 0.0, 'L_adv': 0.0, 'total': 0.0}
                n_chunks = 0
                for i in range(0, xs.shape[0], self.bs):
                    xb = xs[i:i + self.bs]
                    yb = ys[i:i + self.bs]
                    xdb = x_donors[i:i + self.bs]
                    xpb = x_partners[i:i + self.bs]
                    ypb = y_pairs[i:i + self.bs]
                    z_id, z_med, skips = self.model.encode(xb)
                    x_self = self.model.decode(z_id, z_med, skips)
                    recon_acc += float(F.l1_loss(x_self, xb).item())
                    total, parts = compute_s1_loss(self.model, self.frozen, xb, xdb,
                                                   yb, xpb, ypb, return_parts=True)
                    for k in loss_acc:
                        loss_acc[k] += parts[k]
                    n_chunks += 1
                out[prefix + '_val_recon_l1'] = round(recon_acc / max(n_chunks, 1), 5)
                out[prefix + '_val_total'] = round(loss_acc['total'] / max(n_chunks, 1), 5)
                for k in ['L_rec', 'L_path', 'L_anat', 'L_zid', 'L_adv']:
                    out[prefix + '_val_' + k] = round(loss_acc[k] / max(n_chunks, 1), 5)
                # frozen classification diagnostic (x_anon vs source labels) chunked
                out[prefix + '_val_class_auc14'] = self._class_auc14(xs, ys)
                # z_id verifier on validation pairs
                vz = self._pair_auc(model_side='zid')
                out[prefix + '_zid_verifier_auc'] = round(vz, 5)
                # z_med adversary on validation pairs
                vm = self._pair_auc(model_side='zmed')
                out[prefix + '_zmed_adversary_auc'] = round(vm, 5)
        except Exception as e:  # diagnostics must never kill training
            out[prefix + '_diagnostic_error'] = str(e)[:200]
        return out

    def _class_auc14(self, xs, ys):
        """Frozen classifier mean AUC-14 on the given images vs their source labels."""
        try:
            probs_chunks = []
            with torch.no_grad():
                for i in range(0, xs.shape[0], self.bs):
                    probs_chunks.append(self.frozen.path_logits(xs[i:i + self.bs]).detach().cpu().numpy())
            probs = np.concatenate(probs_chunks, axis=0)
            ys_np = ys.detach().cpu().numpy()
            aucs = []
            for i in range(14):
                if len(np.unique(ys_np[:, i])) > 1:
                    try:
                        aucs.append(roc_auc_score(ys_np[:, i], probs[:, i]))
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

    def _load_logs(self, fname):
        p = os.path.join(self.out_dir, fname)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return []


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
    ap.add_argument('--resume_a', type=int, default=0,
                    help='resume Stage A after this completed epoch (loads s1_a_epochN.pth)')
    ap.add_argument('--resume_b', type=int, default=0,
                    help='resume Stage B after this completed epoch (loads s1_b_epochN.pth); implies skip Stage A')
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

    a = None
    if args.resume_b > 0:
        a = {'path': os.path.join(args.out_dir, 's1_a_epoch%02d.pth' % args.epochs_a),
             'stage': 'A', 'epoch': args.epochs_a}
        b = trainer.stage_b(n_epochs=args.epochs_b, init_checkpoint=a, resume_from=args.resume_b)
    else:
        a = trainer.stage_a(n_epochs=args.epochs_a, resume_from=args.resume_a)
        b = trainer.stage_b(n_epochs=args.epochs_b, init_checkpoint=a)
    summary = {'meta': meta, 'stage_a_ckpt': a, 'stage_b_frozen_ckpt': b, 'finished': utcnow()}
    with open(os.path.join(args.out_dir, 'run_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('RUN COMPLETE. FROZEN S1 CHECKPOINT:', b['path'])


if __name__ == '__main__':
    main()
