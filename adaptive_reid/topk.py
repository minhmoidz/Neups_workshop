"""Top-k frozen gallery/probe infrastructure (STEP 2B Part 10).

Canonical configuration:
    N = 500 patients
    fixed selection seed = 42
    CLEAN gallery  (real image)
    ANONYMIZED probe (deformed image)

Metadata source: chexnet/nih_labels.csv -- columns 'Image Index', 'Follow-up #',
'Patient ID', 'fold' (verified present).

The SAME frozen list file must be reused by every arm.
"""

import os

import numpy as np
import pandas as pd

from . import constants as C

FROZEN_LIST_FILENAME = 'topk_frozen_list.csv'


def load_test_metadata(metadata_path='chexnet/nih_labels.csv'):
    df = pd.read_csv(metadata_path)
    required = {'Image Index', 'Follow-up #', 'Patient ID', 'fold'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError('metadata missing required columns: %r' % sorted(missing))
    if 'Image Index' not in df.columns or 'Follow-up #' not in df.columns:
        raise ValueError('metadata must provide Follow-up # to pick gallery/probe by follow-up')
    return df


def build_frozen_topk_list(n_patients=C.TOPK_N_PATIENTS, seed=C.TOPK_SELECTION_SEED,
                           metadata_path='chexnet/nih_labels.csv'):
    """Build the frozen gallery/probe list for N patients (>=2 test images each).

    Prefers gallery/probe images with DIFFERENT follow-up numbers where available;
    falls back to any distinct test images otherwise. Selection is fully seeded so the
    list is reproducible and shared across arms.

    :return: pandas.DataFrame with columns:
        patient_id, gallery_image, gallery_followup, probe_image, probe_followup
    """
    df = load_test_metadata(metadata_path)
    df = df[df['fold'] == 'test']

    sizes = df.groupby('Patient ID').size()
    eligible = sizes[sizes >= 2]
    rng = np.random.RandomState(seed)
    n = min(n_patients, len(eligible))
    patients = list(eligible.sample(n, random_state=seed).index)

    records = []
    for pid in patients:
        imgs = df[df['Patient ID'] == pid][['Image Index', 'Follow-up #']].reset_index(drop=True)
        followups = imgs['Follow-up #'].astype(int).values
        if len(set(followups)) >= 2:
            # prefer a pair with different follow-up numbers
            contenders = []
            for i in range(len(imgs)):
                for j in range(len(imgs)):
                    if i != j and followups[i] != followups[j]:
                        contenders.append((i, j))
            i, j = contenders[rng.randint(len(contenders))]
        else:
            two = rng.choice(len(imgs), size=2, replace=False)
            i, j = int(two[0]), int(two[1])
        records.append({
            'patient_id': pid,
            'gallery_image': imgs.loc[i, 'Image Index'],
            'gallery_followup': imgs.loc[i, 'Follow-up #'],
            'probe_image': imgs.loc[j, 'Image Index'],
            'probe_followup': imgs.loc[j, 'Follow-up #'],
        })
    return pd.DataFrame(records)


def save_frozen_topk_list(path, df, metadata_path='chexnet/nih_labels.csv'):
    """Persist the frozen list (matching the frozen protocol layout)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if not {'patient_id', 'gallery_image', 'gallery_followup',
            'probe_image', 'probe_followup'}.issubset(set(df.columns)):
        raise ValueError('frozen list requires patient_id/gallery_image/gallery_followup/'
                         'probe_image/probe_followup columns')
    df.to_csv(path, index=False)
    return path


def load_frozen_topk_list(path):
    return pd.read_csv(path)