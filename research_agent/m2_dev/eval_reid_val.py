"""M1.2 scientific VAL privacy metric evaluator.

The TRAIN/VAL analogue of upstream utils.test_snn()'s realistic linkage threat
model — evaluated on the fixed VALIDATION pair file:

    anon(image1), real(image2)

The second image is UNTOUCHED by the anonymizer (real image attempted to be
linked to an anonymized image). Produces y_true, y_score and ROC-AUC.

This is the metric that becomes AUC_Bdev_VAL / AUC_C4_VAL for the M1.1 S1 gate.

It is NOT the attacker checkpoint-selection metric (that is BCE loss on
anon/anon validation, see dev_attacker.validate_selection).
"""
import torch
import numpy as np
from sklearn import metrics

from .evaluator_common import snn_preprocess, firewall_check


def evaluate_reid_val_mixed(anonymize_fn, attacker_net, validation_loader, device=None):
    """Evaluate the frozen attacker under the anon/real VALIDATION threat model.

    :param anonymize_fn: shared legacy anonymizer callable (frozen generator).
    :param attacker_net: best (frozen) attacker SiameseNetwork.
    :param validation_loader: fixed VALIDATION pair loader (never a TEST loader).
    :param device: torch device.
    :return: dict with y_true (np), y_score (np), roc_auc (float),
        plus optional threshold-dependent diagnostics (not promotion metrics).
    """
    firewall_check('dev')
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attacker_net = attacker_net.to(device)
    attacker_net.eval()

    y_true = []
    y_score = []

    with torch.no_grad():
        for inputs1, inputs2, labels in validation_loader:
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

            # Scientific VAL privacy geometry: anon(x1), real(x2) untouched.
            x1 = anonymize_fn(inputs1)
            x2 = inputs2

            x1, x2 = snn_preprocess(x1), snn_preprocess(x2)
            outputs = torch.sigmoid(attacker_net(x1, x2)).squeeze()

            y_true.append(labels.cpu().numpy())
            y_score.append(outputs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)

    roc_auc = float(metrics.roc_auc_score(y_true, y_score))

    # Optional diagnostics — recorded but NOT promotion metrics.
    y_pred = (y_score > 0.5).astype(int)
    accuracy = float(metrics.accuracy_score(y_true, y_pred))
    precision = float(metrics.precision_score(y_true, y_pred, zero_division=0))
    recall = float(metrics.recall_score(y_true, y_pred, zero_division=0))
    f1 = float(metrics.f1_score(y_true, y_pred, zero_division=0))

    return {
        'y_true': y_true,
        'y_score': y_score,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }