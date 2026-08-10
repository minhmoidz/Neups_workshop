"""Validation metrics helpers for the adaptive re-ID protocol.

Pure-Python / numpy implementations so they are unit-testable without a GPU. The
canonical scores are continuous logits (or sigmoid probabilities); ROC-AUC is always
computed on continuous scores, never on thresholded predictions. Accuracy uses the
canonical binary threshold 0.5 (on the probability scale).
"""

import numpy as np
from sklearn import metrics as sk_metrics


def continuous_scores(scores):
    """Validate and return a contiguous float64 numpy array of continuous scores.

    :raises ValueError: if the input is empty, contains NaN/Inf, or is not 1-D.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("scores must be 1-D, got shape %r" % (arr.shape,))
    if arr.size == 0:
        raise ValueError("scores array is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores contain NaN or Inf")
    return arr


def binary_truth(labels):
    """Validate and return a contiguous int numpy array of binary labels.

    :raises ValueError: if labels are not binary (only 0/1 allowed), empty, or not 1-D.
    """
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError("labels must be 1-D, got shape %r" % (arr.shape,))
    if arr.size == 0:
        raise ValueError("labels array is empty")
    arr = arr.astype(np.int64)
    if set(np.unique(arr)) - {0, 1}:
        raise ValueError("labels must be binary (0/1)")
    return arr


def check_aligned(scores, labels):
    scores = continuous_scores(scores)
    labels = binary_truth(labels)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels length mismatch: %d vs %d" %
                         (scores.shape[0], labels.shape[0]))
    return scores, labels


def compute_auc(scores, labels):
    """ROC-AUC on continuous scores.

    Perfect separation -> 1.0, reversed -> 0.0, all-tied scores -> 0.5 (sklearn returns
    0.5 for constant scores and warns). A single-class input raises (AUC undefined).
    """
    scores, labels = check_aligned(scores, labels)
    if len(np.unique(labels)) < 2:
        raise ValueError("ROC-AUC is undefined for a single-class label vector")
    return float(sk_metrics.roc_auc_score(labels, scores))


def compute_accuracy(scores, labels, threshold=0.5):
    """Binary accuracy at the canonical 0.5 threshold on probability-scale scores."""
    scores, labels = check_aligned(scores, labels)
    preds = (scores > threshold).astype(np.int64)
    return float(np.mean(preds == labels))


def validation_metrics(scores, labels, threshold=0.5):
    """Return {'auc': float, 'accuracy': float} for a validation partition."""
    return {
        'auc': compute_auc(scores, labels),
        'accuracy': compute_accuracy(scores, labels, threshold=threshold),
    }