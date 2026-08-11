import json
import hashlib
import subprocess
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import torch

LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
          'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def git_head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

def main():
    aucs = pd.read_csv('research_agent/03H_artifacts/classification/aucs.csv').set_index('label')
    preds = pd.read_csv('research_agent/03H_artifacts/classification/preds.csv')
    nih = pd.read_csv('chexnet/nih_labels.csv')
    test = nih[nih['fold'] == 'test'][['Image Index'] + LABELS]
    m = preds.merge(test, on='Image Index')
    assert len(m) == 25596, len(m)

    auc_by_label = {}
    for L in LABELS:
        auc_by_label[L] = float(aucs.loc[L, 'auc'])
    mean_auc_14 = float(np.mean(list(auc_by_label.values())))

    rng = np.random.default_rng(42)
    n = len(m)
    def boot():
        idx = rng.integers(0, n, n)
        return float(np.mean([roc_auc_score(m[L].values[idx].astype(int), m['prob_' + L].values[idx]) for L in LABELS]))
    boot_vals = np.array([boot() for _ in range(500)])
    ci = [float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5))]

    gen = sha256('networks/corrected_baseline/generator_lowest_total_loss_corrected.pth')
    clf = sha256('networks/pretrained_classifier.pth')
    split = sha256('chexnet/nih_labels.csv')
    config = sha256('config_files/config_eval_classifier_corrected_baseline.json')
    eval_script = sha256('eval_classifier.py')
    eval_model = sha256('chexnet/eval_model.py')

    out = {
        'generator_path': 'networks/corrected_baseline/generator_lowest_total_loss_corrected.pth',
        'generator_sha256': gen,
        'classifier_checkpoint': './networks/pretrained_classifier.pth',
        'classifier_sha256': clf,
        'classifier_architecture': 'DenseNet-121 (torchvision), 14-output sigmoid head, frozen at eval',
        'classifier_trained_on': 'raw (unperturbed) NIH ChestX-ray14 images (fold=train)',
        'dataset_split': 'chexnet/nih_labels.csv fold==test',
        'dataset_split_sha256': split,
        'transform_mode': 'corrected',
        'mu': 0.01,
        'stochastic_lambda': 0.0,
        'auc_by_label': auc_by_label,
        'n_positive_negative': {
            L: {'n_positive': int((test[L] > 0).sum()), 'n_negative': int((test[L] <= 0).sum())} for L in LABELS
        },
        'mean_auc_14': mean_auc_14,
        'bootstrap_95ci_mean_auc_14': ci,
        'n_cases': int(len(m)),
        'evaluation_timestamp': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_head(),
        'eval_config_path': 'config_files/config_eval_classifier_corrected_baseline.json',
        'eval_config_sha256': config,
        'eval_script_sha256': eval_script,
        'eval_model_sha256': eval_model,
        'pred_label_arrays_preserved': 'research_agent/03H_artifacts/classification/preds.csv (+ labels joined from nih_labels.csv)',
    }
    with open('research_agent/03H_corrected_classification.json', 'w') as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print('mean_auc_14 = %.6f  CI=[%.4f, %.4f]' % (mean_auc_14, *ci))
    print('wrote research_agent/03H_corrected_classification.json')

if __name__ == '__main__':
    main()
