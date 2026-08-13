"""STEP 7F — Deterministic Subset Selector

Selects exactly 48 TRAIN and 48 VALIDATION images from the NIH dataset.
Seed = 42. Maximizes coverage of the 14 NIH pathology labels.
No TEST access.
"""

import hashlib
import json
import os
import random
import numpy as np
import pandas as pd

LABELS_CSV = 'chexnet/nih_labels.csv'
TRAIN_IMAGES_JSON = 'research_agent/ibr_s1_condition_cache/train_images.json'
VAL_IMAGES_JSON = 'research_agent/ibr_s1_condition_cache/val_images.json'
DATA_DIRS = [
    '/home/minhtt/datasets/nih/images',
    'data',
    'data/images',
]
NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]


def find_image_path(img_id):
    img_name = os.path.basename(img_id)
    for d in DATA_DIRS:
        p = os.path.join(d, img_name)
        if os.path.exists(p):
            return p
    if os.path.exists(img_id):
        return img_id
    return None


def select_subset(candidate_images, labels_df, n_samples=48, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    # Filter candidates to only those existing in labels_df and found on disk
    valid_candidates = []
    for img in candidate_images:
        img_id = os.path.basename(img)
        if img_id in labels_df.index:
            p = find_image_path(img_id)
            if p is not None:
                valid_candidates.append(img_id)
                
    valid_candidates = sorted(list(set(valid_candidates)))
    sub_df = labels_df.loc[valid_candidates]

    
    # Greedy selection to maximize coverage across the 14 labels
    selected = []
    label_counts = {lbl: 0 for lbl in NIH_LABELS}
    
    # First, pick positive examples for each label to ensure both classes represented
    rng = np.random.default_rng(seed)
    shuffled_labels = list(NIH_LABELS)
    rng.shuffle(shuffled_labels)
    
    remaining = set(valid_candidates)
    
    # Pass 1: Try to pick at least 2 positives for each label
    for target_count in [1, 2, 3]:
        for lbl in shuffled_labels:
            if len(selected) >= n_samples:
                break
            if label_counts[lbl] < target_count:
                candidates_for_lbl = [img for img in remaining if sub_df.loc[img, lbl] == 1]
                if candidates_for_lbl:
                    # Pick candidate that brings most new label positives
                    best_c = max(candidates_for_lbl, key=lambda c: sum(sub_df.loc[c, l] for l in NIH_LABELS if label_counts[l] == 0))
                    selected.append(best_c)
                    remaining.remove(best_c)
                    for l in NIH_LABELS:
                        if sub_df.loc[best_c, l] == 1:
                            label_counts[l] += 1

    # Pass 2: Pick "No Finding" / normal cases (all zeros) to balance negatives
    normal_candidates = [img for img in remaining if sub_df.loc[img, NIH_LABELS].sum() == 0]
    num_normals_to_add = min(n_samples // 4, len(normal_candidates))
    selected_normals = rng.choice(normal_candidates, size=num_normals_to_add, replace=False)
    for c in selected_normals:
        if len(selected) < n_samples and c in remaining:
            selected.append(c)
            remaining.remove(c)
            
    # Pass 3: Fill remaining slots deterministically
    sorted_remaining = sorted(list(remaining))
    rng.shuffle(sorted_remaining)
    while len(selected) < n_samples and sorted_remaining:
        c = sorted_remaining.pop(0)
        selected.append(c)
        for l in NIH_LABELS:
            if sub_df.loc[c, l] == 1:
                label_counts[l] += 1
                
    assert len(selected) == n_samples, f"Selected {len(selected)} != {n_samples}"
    assert len(set(selected)) == n_samples, "Duplicates detected in selection"
    
    # Compile records
    records = []
    for idx, img_id in enumerate(selected):
        row = sub_df.loc[img_id]
        img_path = find_image_path(img_id)
        records.append({
            'index': idx,
            'image_id': img_id,
            'image_path': img_path,
            'labels': {lbl: int(row[lbl]) for lbl in NIH_LABELS},
            'positive_labels': [lbl for lbl in NIH_LABELS if row[lbl] == 1],
        })
        
    return records, label_counts


def build_manifest(out_path='research_agent/compdiff_artifacts/subset_manifest.json'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    labels_df = pd.read_csv(LABELS_CSV).set_index('Image Index')
    
    with open(TRAIN_IMAGES_JSON) as f:
        train_candidates = json.load(f)
    with open(VAL_IMAGES_JSON) as f:
        val_candidates = json.load(f)
        
    train_records, train_counts = select_subset(train_candidates, labels_df, n_samples=48, seed=42)
    val_records, val_counts = select_subset(val_candidates, labels_df, n_samples=48, seed=42)
    
    # Verify no overlap between train and val subsets
    train_set = {r['image_id'] for r in train_records}
    val_set = {r['image_id'] for r in val_records}
    assert len(train_set.intersection(val_set)) == 0, "Train and Validation subsets overlap!"
    
    manifest = {
        'seed': 42,
        'train_count': len(train_records),
        'val_count': len(val_records),
        'nih_labels': NIH_LABELS,
        'train_label_distribution': train_counts,
        'val_label_distribution': val_counts,
        'train_evaluable_labels': [lbl for lbl, c in train_counts.items() if 0 < c < 48],
        'val_evaluable_labels': [lbl for lbl, c in val_counts.items() if 0 < c < 48],
        'train_subset': train_records,
        'val_subset': val_records,
    }
    
    manifest_bytes = json.dumps(manifest, indent=2).encode('utf-8')
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest['manifest_sha256'] = manifest_sha
    
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Manifest written to {out_path} (SHA-256: {manifest_sha})")
    print(f"TRAIN evaluable labels ({len(manifest['train_evaluable_labels'])}/14): {manifest['train_evaluable_labels']}")
    print(f"VAL evaluable labels ({len(manifest['val_evaluable_labels'])}/14): {manifest['val_evaluable_labels']}")
    return manifest
