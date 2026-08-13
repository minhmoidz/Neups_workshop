"""Phase-II IBR S1 - deterministic donor sampler (STEP 6A lock #5).

Protocol (locked in STEP 6A):
    - donor patient != source patient (hard assertion)
    - TRAIN source  -> TRAIN donor only
    - VALIDATION source -> allowed validation donor pool only (patient-disjoint rules)
    - no TEST donor pool during development
    - deterministic donor mapping given a fixed seed; changing seed changes mapping
    - persist donor/source patient IDs in diagnostic output
"""

import json
import hashlib
import os

import numpy as np
import pandas as pd

NIH_LABELS_PATH = "chexnet/nih_labels.csv"


class DonorSampler:
    """Deterministic donor patient selection per split.

    Builds a patient->image map from nih_labels.csv. Given a batch of source
    image indices (each carrying its Patient ID), selects for every source a
    different-patient donor image from the allowed pool for that split.

    Allowed donor pools:
        'train'       -> train patient images only
        'validation'  -> validation patient images only
    TEST is never allowed as a donor pool.

    Deterministic: donor choice is a pure function of (seed, source images).
    """

    def __init__(self, seed=0):
        self.seed = int(seed)
        self.df = pd.read_csv(NIH_LABELS_PATH, usecols=['Image Index', 'Patient ID', 'fold'])
        self.fold_by_image = dict(zip(self.df['Image Index'], self.df['fold']))
        self.patient_by_image = dict(zip(self.df['Image Index'], self.df['Patient ID']))
        self._images_by_patient = {}
        for _, row in self.df.iterrows():
            self._images_by_patient.setdefault(int(row['Patient ID']), []).append(row['Image Index'])
        # per-split pool: patient_id -> list of images, plus patient list
        self._pool = {}
        for fold in ['train', 'validation', 'test']:
            if fold == 'validation':
                sub = self.df[self.df['fold'] == 'val']
            else:
                sub = self.df[self.df['fold'] == fold]
            self._pool[fold] = {
                'patients': sorted(sub['Patient ID'].unique().astype(int).tolist()),
                'images_by_patient': {},
            }
            for _, row in sub.iterrows():
                self._pool[fold]['images_by_patient'].setdefault(int(row['Patient ID']), []).append(row['Image Index'])

    # -- split resolution ---------------------------------------------------
    def _split_for(self, source_image):
        fold = self.fold_by_image[source_image]
        if fold == 'val':
            return 'validation'
        return fold

    def allowed_donor_pool(self, source_images):
        """Return the single allowed donor pool name for a set of source images.

        All sources in one batch come from the same split; enforce consistency.
        TEST sources are forbidden (no TEST donor pool in development).
        """
        splits = {self._split_for(s) for s in source_images}
        if 'test' in splits:
            raise RuntimeError("TEST source images are not allowed in Phase-II development (no TEST donor pool).")
        if len(splits) > 1:
            raise RuntimeError("Mixed-split source batch is not supported: %s" % sorted(splits))
        return splits.pop()

    # -- deterministic donor selection --------------------------------------
    def donor_for(self, source_image):
        """Deterministic per-source donor (pure function of seed + image name).

        Unlike __call__ (which keys on the sorted batch list), this gives an
        identical donor for a source image regardless of how it is batched, so
        donor loading can be parallelised inside DataLoader workers while the
        mapping stays reproducible across processes and batchings.
        """
        src_pid = int(self.patient_by_image[source_image])
        split = self._split_for(source_image)
        pool = self._pool[split]
        candidates = []
        for pid in pool['patients']:
            if pid != src_pid:
                candidates.extend(pool['images_by_patient'][pid])
        assert len(candidates) > 0, "No donor candidate for %s in %s" % (source_image, split)
        key = hashlib.sha256(repr(source_image).encode()).hexdigest()
        rng = np.random.default_rng(self.seed + int(key[:16], 16) % (2**32))
        donor = candidates[int(rng.integers(0, len(candidates)))]
        assert self.patient_by_image[donor] != src_pid, "Donor must differ from source patient"
        return donor

    def __call__(self, source_images, rng=None):
        """Return donor image index (str) for each source image index.

        Deterministic for a fixed seed: derive per-sample draw positions from a
        seeded RNG keyed on the sorted source list.
        """
        source_images = list(source_images)
        if rng is None:
            # Content-addressed key: built-in hash() is salted per-process, which would
            # break cross-process reproducibility. Use a stable digest of the sorted
            # source list (STEP 6A lock #5: deterministic donor mapping given a seed,
            # reproducible across runs).
            key = hashlib.sha256(repr(sorted(source_images)).encode()).hexdigest()
            rng = np.random.default_rng(self.seed + int(key[:16], 16) % (2**32))
        split = self.allowed_donor_pool(source_images)
        pool = self._pool[split]
        donor_images = []
        for src in source_images:
            src_pid = int(self.patient_by_image[src])
            # candidates: images of patients different from source
            candidates = []
            for pid in pool['patients']:
                if pid != src_pid:
                    candidates.extend(pool['images_by_patient'][pid])
            assert len(candidates) > 0, "No donor candidate for %s in %s" % (src, split)
            donor = candidates[int(rng.integers(0, len(candidates)))]
            assert self.patient_by_image[donor] != self.patient_by_image[src], (
                "Donor must differ from source patient")
            donor_images.append(donor)
        return donor_images

    def provenance(self, source_images, donor_images):
        return [{'source': s, 'source_patient': int(self.patient_by_image[s]),
                 'donor': d, 'donor_patient': int(self.patient_by_image[d])}
                for s, d in zip(source_images, donor_images)]


def build_donor_map_csv(source_images, donor_images, sampler, out_path):
    recs = sampler.provenance(source_images, donor_images)
    with open(out_path, 'w') as f:
        json.dump(recs, f, indent=2)
    return out_path