"""P0 reproduction-only scientific attacker training loop (P0_2_3).

Implements the locked canonical attacker protocol against a frozen generator:

  - Siamese ResNet-50, fresh ImageNet init per seed (weights from the
    human-approved local artifact; never downloaded here);
  - Adam lr 1e-4, batch 32, max 100 epochs, patience 5;
  - TRAIN/selection geometry: anon(x1)/anon(x2) with the generator in
    .eval() under torch.no_grad() via the P0 state guard;
  - final scoring geometry: anon(x1)/real(x2), sigmoid scores,
    sklearn ROC AUC (raw; direction fixed a priori);
  - TRAIN order exclusively from the P0_SAMPLER_V1_1 deterministic sampler;
  - outputs bound to bytes: predictions.csv + attacker_best.pth + sealed
    run manifest.

Importing this module has NO side effects and does not touch CUDA. All heavy
imports happen inside functions. Governed source is used read-only.
"""
import copy
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

# Process-level cache of eagerly-decoded, immutable datasets shared across
# attacker seeds within one screen/bridge process (see _real_dataset_factory).
_DATASET_CACHE = {}


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_anonymize_from_generator(generator, protocol, mu=None):
    """Frozen legacy flow-field operator bound to an already-loaded generator.

    Mirrors evaluator constants: grid = identity - mu*grid; GaussianSmoothing
    kernel/sigma from the locked protocol; grid_sample border/align_corners.
    """
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    sys_path = os.path.abspath(os.path.join(HERE, "..", ".."))
    if sys_path not in __import__("sys").path:
        __import__("sys").path.insert(0, sys_path)
    from utils.GaussianSmoothing import GaussianSmoothing

    ap = protocol["attacker_protocol"]["flow_operator"]
    # P0_PROTOCOL_V1_2: mu is a property of the GENERATOR being evaluated, not
    # of the attacker. The arm's value wins; the V1_1 global remains the
    # default so every V1_1 arm resolves to exactly its V1_1 value.
    mu = float(ap["mu"] if mu is None else mu)
    k = int(ap["gaussian_kernel"]); sigma = int(ap["gaussian_sigma"])
    dev = next(generator.parameters()).device
    cache = {}
    gauss_cache = {}

    def anonymize(image):
        size = image.shape[-1]
        if size not in cache:
            d = torch.linspace(-1, 1, size)
            mesh_x, mesh_y = torch.meshgrid((d, d), indexing="ij")
            grid_identity = torch.stack((mesh_y, mesh_x), 2).unsqueeze(0)\
                .permute(0, 3, 1, 2).to(dev)
            cache[size] = grid_identity
        if (k, sigma) not in gauss_cache:
            gauss_cache[(k, sigma)] = GaussianSmoothing(
                channels=2, kernel_size=k, sigma=sigma).to(dev)
        with torch.no_grad():
            grids = generator(image)
            grids = cache[size] - mu * grids
            grids = gauss_cache[(k, sigma)](grids)
            grids = grids.permute(0, 2, 3, 1)
            return F.grid_sample(image, grids, padding_mode="border",
                                 align_corners=True)
    return anonymize


class P0AttackerRunner:
    """One paired attacker trajectory for ONE arm and master seed."""

    def __init__(self, *, arm, master_seed, protocol, repo_root,
                 image_root=None, device=None, dataset_factory=None,
                 net_factory=None, generator_factory=None,
                 protocol_sha256=None, runner_commit=None):
        self.protocol_sha256 = protocol_sha256
        self.runner_commit = runner_commit
        sys_mod = __import__("sys")
        for pth in (os.path.abspath(os.path.join(HERE, "..", "..")),
                    os.path.abspath(os.path.join(HERE, "..", "..", "research_agent"))):
            if pth not in sys_mod.path:
                sys_mod.path.insert(0, pth)

        self.arm = arm
        self.master_seed = int(master_seed)
        self.protocol = protocol
        self.repo_root = repo_root
        ap = protocol["attacker_protocol"]
        self.batch_size = int(ap["batch_size"])
        self.max_epochs = int(ap["max_epochs"])
        self.patience = int(ap["early_stopping_patience"])

        from seed_contract import (
            derive_seed, seed_everything_for_attacker_construction)
        self.derived_seeds = {d: derive_seed(self.master_seed, d)
                              for d in ap["domains"]}
        # weight seeding IMMEDIATELY before attacker construction
        seed_everything_for_attacker_construction(
            self.derived_seeds["attacker_weight_init"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        arm_spec = protocol["arms"][arm]
        exec_path = os.path.join(repo_root, arm_spec["execution_path"])
        if generator_factory is not None:
            self.generator = generator_factory().to(self.device)
        else:
            from networks.UNet_PriCheXyNet import UNet
            self.generator = UNet(1, 2, 32).to(self.device)
        self.generator.load_state_dict(torch.load(exec_path, map_location=self.device,
                                                  weights_only=False))
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad_(False)
        self.resolved_mu = float(arm_spec.get(
            "mu", protocol["attacker_protocol"]["flow_operator"]["mu"]))
        self.anonymize = build_anonymize_from_generator(self.generator,
                                                        protocol,
                                                        mu=self.resolved_mu)

        from networks.SiameseNetwork import SiameseNetwork
        self.net = (net_factory() if net_factory is not None
                    else SiameseNetwork()).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=float(ap["learning_rate"]))

        if dataset_factory is None:
            dataset_factory = self._real_dataset_factory(image_root)
        self._dataset_factory = dataset_factory

        from deterministic_sampler import make_paired_dataloader
        self.train_loader_builder = make_paired_dataloader(
            dataset_factory("train"), self.master_seed,
            num_workers=ap.get("num_workers", 0),
            batch_size=self.batch_size)
        val_ds = dataset_factory("val")
        self.val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False)

        from deterministic_sampler import order_hash, build_permutation
        self._order_hash = lambda e: order_hash(
            build_permutation(self.master_seed, e, len(dataset_factory("train"))),
            self.master_seed, e, len(dataset_factory("train")))

        from generator_guard import canonical_model_state_hash
        self.initial_state_hash = canonical_model_state_hash(self.net)
        self.started_utc = time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                         time.gmtime())
        self.loss_history = {"training": [], "validation": []}
        self.epoch_order_hashes = {}
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.stop_epoch = None
        self.best_state = None

    def _real_dataset_factory(self, image_root):
        """Build or REUSE a cached eager pair dataset (P0_2_3 perf).

        LazyPairDataset decodes ~12k images ONCE in __init__; sharing one
        immutable instance across attacker seeds within this process removes
        repeated decode cost without changing any tensor, label, or order —
        the dataset is read-only after construction, and ordering comes
        exclusively from the deterministic sampler.
        """
        def factory(phase):
            from m2_dev.evaluator_common import LazyPairDataset
            phase_name = "training" if phase == "train" else "validation"
            key = (phase_name, os.path.abspath(image_root))
            if key not in _DATASET_CACHE:
                _DATASET_CACHE[key] = LazyPairDataset(
                    phase=phase_name, image_path=image_root,
                    metadata_path=os.path.join(
                        self.repo_root, "Data_Entry_2017_v2020.csv"))
            return _DATASET_CACHE[key]
        # LazyPairDataset reads its own fixed pair files by phase; the
        # protocol-locked paths are byte-identical to those files (verified).
        return factory

    # ------------------------------------------------------------------
    def _anon_pair_train(self, x1, x2):
        from m2_dev.evaluator_common import snn_preprocess
        x1a = self.anonymize(x1)
        x2a = self.anonymize(x2)
        return snn_preprocess(x1a), snn_preprocess(x2a)

    def train_epoch(self, loader):
        self.net.train()
        running, n = 0.0, 0
        for batch in loader:
            x1, x2, _, labels_id = batch
            x1, x2 = x1.to(self.device), x2.to(self.device)
            labels_id = labels_id.float().to(self.device)
            x1p, x2p = self._anon_pair_train(x1, x2)
            self.optimizer.zero_grad()
            out = self.net(x1p, x2p).squeeze()
            loss = self.criterion(out, labels_id.type_as(out))
            if not torch.isfinite(loss).all():
                raise FloatingPointError("attacker loss non-finite")
            loss.backward()
            self.optimizer.step()
            running += float(loss.item()); n += 1
        return running / max(n, 1)

    @torch.no_grad()
    def selection_epoch(self, loader):
        """Selection geometry anon/anon — NOT the privacy metric."""
        self.net.eval()
        running, n = 0.0, 0
        for batch in loader:
            x1, x2, _, labels_id = batch
            x1, x2 = x1.to(self.device), x2.to(self.device)
            x1p, x2p = self._anon_pair_train(x1, x2)
            out = self.net(x1p, x2p).squeeze()
            labels_id = labels_id.to(self.device)
            loss = self.criterion(out, labels_id.type_as(out))
            if not torch.isfinite(loss).all():
                raise FloatingPointError("selection loss non-finite")
            running += float(loss.item()); n += 1
        return running / max(n, 1)

    def run_training(self):
        patience_left = self.patience
        for epoch in range(self.max_epochs):
            loader = self.train_loader_builder(epoch)
            tr = self.train_epoch(loader)
            va = self.selection_epoch(self.val_loader)
            self.loss_history["training"].append(tr)
            self.loss_history["validation"].append(va)
            self.epoch_order_hashes[str(epoch)] = self._order_hash(epoch)
            improved = va < self.best_val_loss
            if improved:
                self.best_val_loss = va
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(
                    self.net.state_dict())
                patience_left = self.patience
            else:
                patience_left -= 1
            self.stop_epoch = epoch
            print("[P0 %s seed %d] epoch %d train %.4f sel %.4f%s"
                  % (self.arm, self.master_seed, epoch, tr, va,
                     " *" if improved else ""))
            if patience_left <= 0:
                break
        self.stop_epoch = min(self.stop_epoch, self.max_epochs - 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_anon_real(self):
        """Final scoring geometry anon(x1)/real(x2); raw ROC AUC."""
        from m2_dev.evaluator_common import snn_preprocess
        from sklearn.metrics import roc_auc_score
        self.net.eval()
        self.net.load_state_dict(self.best_state)
        y_true, y_score, pair_idx = [], [], []
        for bi, batch in enumerate(self.val_loader):
            x1, x2, _, labels_id = batch
            x1, x2 = x1.to(self.device), x2.to(self.device)
            x1p = snn_preprocess(self.anonymize(x1))
            x2p = snn_preprocess(x2)
            scores = torch.sigmoid(self.net(x1p, x2p).squeeze(-1))
            y_true += [int(v) for v in labels_id.cpu().numpy()]
            y_score += [float(v) for v in scores.cpu().numpy()]
            pair_idx += [bi * self.batch_size + i
                         for i in range(len(labels_id))]
        auc = float(roc_auc_score(np.asarray(y_true), np.asarray(y_score)))
        return {"y_true": y_true, "y_score": y_score,
                "pair_index": pair_idx, "raw_roc_auc": auc}

    # ------------------------------------------------------------------
    def finalize(self, runs_root):
        """Fresh claim -> write bytes -> seal manifest. Returns manifest."""
        from manifest_io import (claim_run_directory, canonical_json_bytes,
                                 write_run_manifest_atomic,
                                 RUN_MANIFEST_SCHEMA)
        run_dir = claim_run_directory(runs_root, self.arm, self.master_seed)

        score = self.score_anon_real()
        import pandas as pd
        df = pd.DataFrame({
            "pair_index": score["pair_index"],
            "y_true": score["y_true"],
            "y_score": [round(v, 6) for v in score["y_score"]],
        })
        csv_payload = df.to_csv(index=False).encode("utf-8")
        final_csv = os.path.join(run_dir, "predictions.csv")
        with open(final_csv, "wb") as fh:
            fh.write(csv_payload); fh.flush(); os.fsync(fh.fileno())

        ckpt_path = os.path.join(run_dir, "attacker_best.pth")
        torch.save({"state_dict": self.best_state,
                    "best_epoch": self.best_epoch}, ckpt_path)

        manifest = {
            "schema_version": RUN_MANIFEST_SCHEMA,
            "protocol_schema": self.protocol["schema_version"],
            "protocol_sha256": self.protocol_sha256,
            "runner_commit": self.runner_commit,
            "arm": self.arm,
            "master_seed": self.master_seed,
            "derived_seeds": dict(self.derived_seeds),
            "generator_role": self.arm,
            "generator_sha256":
                self.protocol["arms"][self.arm]["generator_sha256"],
            # P0_PROTOCOL_V1_2: bind every result to the mu it was ACTUALLY
            # evaluated at. Without this a reader cannot tell whether a
            # mu=0.001 checkpoint was scored with a mu=0.001 deformation.
            "resolved_mu": self.resolved_mu,
            "train_pair_sha256":
                self.protocol["pair_files"]["train"]["sha256"],
            "val_pair_sha256":
                self.protocol["pair_files"]["val"]["sha256"],
            "initial_attacker_state_hash": self.initial_state_hash,
            "epoch_order_hashes": dict(self.epoch_order_hashes),
            "attacker_best_sha256": _sha256_file(ckpt_path),
            "best_epoch": self.best_epoch,
            "stop_epoch": self.stop_epoch,
            "score_direction":
                self.protocol["attacker_protocol"]["score_direction"],
            "predictions_sha256": hashlib.sha256(csv_payload).hexdigest(),
            "environment_provenance": {
                "torch": torch.__version__,
                "device": str(self.device)},
            "status": "COMPLETE",
            "started_utc": self.started_utc,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                          time.gmtime()),
            "raw_roc_auc": score["raw_roc_auc"],
            "best_val_loss": self.best_val_loss,
        }
        write_run_manifest_atomic(run_dir, manifest)
        return run_dir, manifest
