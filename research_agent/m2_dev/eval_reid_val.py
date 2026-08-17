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
import os

import torch
import numpy as np
from sklearn import metrics

from .evaluator_common import (
    snn_preprocess, firewall_check, verify_frozen_val_pair_provenance,
    FROZEN_VAL_PAIR_COUNT, FROZEN_VAL_PAIR_SHA256, SCIENTIFIC_IMAGE_ROOT,
)


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
        for batch in validation_loader:
            # Accepts both the unit-test synthetic 3-tuple (img1, img2, label) and the
            # real LazyPairDataset 4-tuple (img1, img2, ac_labels_1, labels_id): the
            # identity/Re-ID ground-truth label is always the LAST element.
            inputs1, inputs2, labels = batch[0], batch[1], batch[-1]
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

            # Scientific VAL privacy geometry: anon(x1), real(x2) untouched.
            x1 = anonymize_fn(inputs1)
            x2 = inputs2

            x1, x2 = snn_preprocess(x1), snn_preprocess(x2)
            outputs = torch.sigmoid(attacker_net(x1, x2)).squeeze()

            y_true.append(labels.cpu().numpy())
            y_score.append(outputs.cpu().numpy())

    if not y_true or not y_score:
        raise RuntimeError('Privacy VAL evaluator produced no rows')
    y_true = np.asarray(np.concatenate(y_true))
    y_score = np.asarray(np.concatenate(y_score))
    if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
        raise RuntimeError('Privacy evaluator outputs must be equal-length 1-D arrays')
    if not np.isfinite(y_true).all() or not np.isfinite(y_score).all():
        raise FloatingPointError('Privacy evaluator outputs contain NaN/Inf')
    if not np.all(np.isin(y_true, [0, 1])) or len(np.unique(y_true)) != 2:
        raise RuntimeError('Privacy evaluator labels must be binary and contain both classes')
    n_pairs = int(len(y_true))

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
        'n_pairs': n_pairs,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def evaluate_reid_val(config=None, attacker_checkpoint=None, generator_checkpoint=None, device=None,
                      validation_loader=None, unit_test_mode=False, image_size=256,
                      expected_generator_sha=None, expected_attacker_sha=None):
    """End-to-end evaluation helper for scientific VAL privacy.
    Loads generator from generator_checkpoint and attacker from attacker_checkpoint,
    then evaluates on the fixed 2,000 validation pairs under anon/real threat model.
    """
    firewall_check('dev')
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = config or {}
    if not unit_test_mode and getattr(device, 'type', str(device).split(':')[0]) == 'cpu':
        raise RuntimeError('Scientific privacy evaluation requires CUDA; CPU is unit-test only')
    if not unit_test_mode and config.get('image_path') != SCIENTIFIC_IMAGE_ROOT:
        raise RuntimeError('Scientific privacy evaluation requires the approved scientific data root for image_path')

    from .dev_attacker import load_frozen_anonymizer, SiameseNetwork
    from .evaluator_common import build_dev_anonymizer_loaders, file_sha256

    if not generator_checkpoint:
        raise ValueError("generator_checkpoint must be explicitly specified")
    if not attacker_checkpoint:
        raise ValueError("attacker_checkpoint must be explicitly specified")

    if not os.path.exists(generator_checkpoint):
        raise FileNotFoundError("Generator checkpoint not found: %s" % generator_checkpoint)
    if not os.path.exists(attacker_checkpoint):
        raise FileNotFoundError("Attacker checkpoint not found: %s" % attacker_checkpoint)

    gen_sha = file_sha256(generator_checkpoint)
    att_sha = file_sha256(attacker_checkpoint)

    if expected_generator_sha is not None and gen_sha != expected_generator_sha:
        raise RuntimeError("Generator checkpoint SHA mismatch before privacy eval: %s != expected %s" % (gen_sha, expected_generator_sha))
    if expected_attacker_sha is not None and att_sha != expected_attacker_sha:
        raise RuntimeError("Attacker checkpoint SHA mismatch before privacy eval: %s != expected %s" % (att_sha, expected_attacker_sha))

    _, anonymize_fn = load_frozen_anonymizer(device=device, checkpoint_path=generator_checkpoint, image_size=image_size)

    attacker_net = SiameseNetwork().to(device)
    attacker_net.load_state_dict(torch.load(attacker_checkpoint, map_location=device, weights_only=False))

    if not unit_test_mode and validation_loader is not None:
        raise RuntimeError('Scientific privacy evaluation does not accept an injected validation_loader')
    if validation_loader is None:
        validation_loader = config.get('validation_loader')
    if not unit_test_mode and validation_loader is not None:
        raise RuntimeError('Scientific privacy evaluation does not accept config validation_loader injection')

    if validation_loader is None:
        if unit_test_mode:
            # Unit-test mode ALWAYS uses a synthetic loader (never real dataset loaders).
            syn_ds = [(torch.rand(1, image_size, image_size), torch.rand(1, image_size, image_size),
                       torch.tensor(float(i % 2))) for i in range(8)]
            val_loader = torch.utils.data.DataLoader(syn_ds, batch_size=4)
        else:
            _, val_loader, _ = build_dev_anonymizer_loaders(config, seed=42)
    else:
        val_loader = validation_loader

    # Scientific fail-fast: verify the frozen VAL pair provenance BEFORE running.
    pair_provenance = None
    if not unit_test_mode:
        pair_provenance = verify_frozen_val_pair_provenance()
        ds = getattr(val_loader, 'dataset', None)
        n_pairs_known = len(ds) if (ds is not None and hasattr(ds, '__len__')) else None
        if n_pairs_known is not None and n_pairs_known != FROZEN_VAL_PAIR_COUNT:
            raise RuntimeError('Scientific privacy evaluation requires exactly 2000 validation pairs, got %d' % n_pairs_known)

    res = evaluate_reid_val_mixed(anonymize_fn, attacker_net, val_loader, device=device)
    if not unit_test_mode and res['n_pairs'] != FROZEN_VAL_PAIR_COUNT:
        raise RuntimeError('Scientific privacy evaluation requires exactly 2000 validation pairs, got %d' % res['n_pairs'])

    if pair_provenance:
        res.update({
            'validation_pair_file': pair_provenance['pair_file'],
            'validation_pair_file_sha256': pair_provenance['pair_file_sha256'],
            'validation_pair_count': pair_provenance['pair_count'],
            'validation_pair_order_sha256': pair_provenance['pair_order_sha256'],
        })
    res['generator_checkpoint_sha256'] = gen_sha
    res['attacker_checkpoint_sha256'] = att_sha
    return res