"""M1.2 classification VAL-only utility evaluator.

Replaces chexnet/eval_model.make_pred_multilabel() for M2 development because
that function hard-codes test fold and would construct the official TEST
dataset (P0-B). This evaluator:

  - takes an explicit `fold` argument (development fold = "val")
  - REJECTS fold in {test, testing, final_test} BEFORE any dataset
    construction / file read
  - uses the frozen released classifier networks/pretrained_classifier.pth
    (same checkpoint SHA + weights + preprocessing for both arms, §9)
  - applies the M1-frozen legacy flow_field anonymization (frozen generator,
    mu=0.01) to the images before classification
  - returns per-pathology ROC-AUCs and macro_AUC = mean over the 14 classes
"""
import os

import numpy as np
import pandas as pd
import sklearn.metrics as sklm
import torch
from torchvision import transforms

from .evaluator_common import (
    assert_dev_fold,
    firewall_check,
    file_sha256,
    FROZEN_CLASSIFIER_SHA,
    IMAGENET_NORMALIZE,
    classifier_preprocess,
    make_flow_field_components,
    build_anonymize_fn,
    MU,
)

DEV_FOLD = 'val'


def load_frozen_classifier(device=None):
    """Load the released frozen DenseNet-121 classifier and verify its SHA."""
    from networks.UNet_PriCheXyNet import UNet  # noqa: F401  (import guard for compatibility)
    checkpoint = torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         '..', 'networks', 'pretrained_classifier.pth'),
                            map_location='cpu', weights_only=False)
    model = checkpoint['model']
    if device is not None:
        model = model.to(device)
    model.train(False)
    return model


def classify_val_dataset(model, dataloader, anonymize_fn, perturbation_type,
                         device=None, batch_size=16):
    """Run the frozen classifier over a VAL-only dataloader (no TEST fold).

    :return: (pred_df, auc_df, macro_auc)
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    pred_df = pd.DataFrame(columns=['Image Index'])
    true_df = pd.DataFrame(columns=['Image Index'])
    PRED_LABEL = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                  'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                  'Pleural_Thickening', 'Hernia']

    with torch.no_grad():
        for i, (inputs, labels, idx_names) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            if anonymize_fn is not None:
                inputs = anonymize_fn(inputs)

            if perturbation_type in ['flow_field', 'privacy_net', 'dp_pix']:
                inputs = classifier_preprocess(inputs)
            else:
                inputs = inputs.expand(-1, 3, -1, -1)
                inputs = IMAGENET_NORMALIZE(inputs)

            outputs = model(inputs)
            probs = outputs.cpu().data.numpy()
            true_labels = labels.cpu().data.numpy()
            bs = true_labels.shape[0]

            for j in range(bs):
                idx_name = idx_names[j] if isinstance(idx_names, (list, tuple)) else str(idx_names)
                thisrow = {'Image Index': idx_name}
                truerow = {'Image Index': idx_name}
                for k, lbl in enumerate(PRED_LABEL):
                    thisrow['prob_' + lbl] = probs[j, k]
                    truerow[lbl] = true_labels[j, k]
                pred_df = pd.concat([pred_df, pd.DataFrame(thisrow, index=[0])], ignore_index=True)
                true_df = pd.concat([true_df, pd.DataFrame(truerow, index=[0])], ignore_index=True)

    auc_df = pd.DataFrame(columns=['label', 'auc'])
    for column in PRED_LABEL:
        y_true_col = true_df[column].values.astype(int)
        y_score_col = pred_df['prob_' + column].values.astype(float)
        unique_classes = np.unique(y_true_col)
        if len(unique_classes) < 2:
            raise RuntimeError(
                "Classification VAL pathology %r has only %d class in ground truth (cannot compute ROC-AUC)" % (
                    column, len(unique_classes)
                )
            )
        try:
            auc = float(sklm.roc_auc_score(y_true_col, y_score_col))
            if not np.isfinite(auc):
                raise RuntimeError("Classification VAL pathology %r produced non-finite AUC: %s" % (column, auc))
        except Exception as exc:
            raise RuntimeError("Classification VAL pathology %r failed AUC calculation: %s" % (column, exc)) from exc

        auc_df = pd.concat([auc_df, pd.DataFrame({'label': [column], 'auc': [auc]}, index=[0])], ignore_index=True)

    n_valid = len(auc_df['auc'].dropna())
    if n_valid != 14:
        raise RuntimeError("Classification VAL requires exactly 14 valid AUCs, got %d" % n_valid)

    macro_auc = float(auc_df['auc'].mean())
    return pred_df, auc_df, macro_auc


def evaluate_classification_val(config, model=None, fold=DEV_FOLD, device=None,
                                perturbation_type='flow_field', batch_size=16,
                                generator_checkpoint=None, image_size=None,
                                expected_generator_sha=None):
    """VAL-only classification evaluation. fold is validated BEFORE dataset init.

    :param config: dev config dict (image_path, ...).
    :param model: optional injected frozen classifier (tests); defaults to the
        released pretrained_classifier.pth.
    :param fold: development fold; MUST be 'val'.
    :param generator_checkpoint: explicit path to selected M2 generator checkpoint (required for flow_field).
    :param image_size: legacy flow-field grid size; defaults to config image_size or 256.
    :param expected_generator_sha: optional expected SHA256 of the generator checkpoint.
    """
    assert_dev_fold(fold)          # reject TEST before dataset construction
    firewall_check('dev')

    import chexnet.cxr_dataset as CXR

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model is None:
        model = load_frozen_classifier(device)

    # Verify the frozen classifier SHA when using the released checkpoint.
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             '..', 'networks', 'pretrained_classifier.pth')
    if os.path.exists(ckpt_path):
        actual_clf_sha = file_sha256(ckpt_path)
        if actual_clf_sha != FROZEN_CLASSIFIER_SHA:
            raise RuntimeError('classifier checkpoint SHA drift: %s != frozen %s' % (actual_clf_sha, FROZEN_CLASSIFIER_SHA))
    else:
        actual_clf_sha = 'injected_classifier'

    # Anonymizer (frozen generator, legacy operator, mu=0.01)
    from networks.UNet_PriCheXyNet import UNet

    anonymize_fn = None
    selected_gen_sha = None
    if perturbation_type == 'flow_field':
        gen_path = generator_checkpoint
        if not gen_path:
            raise RuntimeError("generator_checkpoint must be explicitly provided (no scientific fallback allowed)")
        if not os.path.exists(gen_path):
            raise FileNotFoundError("Selected generator checkpoint not found: %s" % gen_path)

        selected_gen_sha = file_sha256(gen_path)
        if expected_generator_sha is not None and selected_gen_sha != expected_generator_sha:
            raise RuntimeError("Selected generator SHA mismatch before classification eval: %s != expected %s" % (selected_gen_sha, expected_generator_sha))

        generator = UNet(1, 2, 32).to(device)
        generator.load_state_dict(torch.load(gen_path, map_location=device, weights_only=False))
        if image_size is None:
            image_size = config.get('image_size', 256)
        grid_identity, gauss_filter = make_flow_field_components(device, image_size=image_size)
        anonymize_fn = build_anonymize_fn(generator, grid_identity, gauss_filter, MU)

    if config.get('dataloader') is not None:
        dataloader = config['dataloader']
        n_images = len(dataloader.dataset)
        if not config.get('unit_test_mode', False) and fold == 'val' and n_images != 10816:
            raise RuntimeError("Classification scientific VAL requires exactly 10,816 images, got %d" % n_images)
    elif config.get('unit_test_mode', False):
        from m0_tests.test_m14a_execution_harness import SyntheticClassificationDataset
        if image_size is None:
            image_size = 64
        dataset = SyntheticClassificationDataset(size=28, image_size=image_size)
        n_images = len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        # Build VAL-only dataset + loader (fold must be 'val'; never 'test')
        dataset = CXR.CXRDataset(
            path_to_images=config['image_path'],
            fold=fold,
            transform=None,
            perturbation_type=perturbation_type)
        n_images = len(dataset)
        if not config.get('unit_test_mode', False) and fold == 'val' and n_images != 10816:
            raise RuntimeError("Classification scientific VAL requires exactly 10,816 images, got %d" % n_images)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)

    pred_df, auc_df, macro_auc = classify_val_dataset(
        model, dataloader, anonymize_fn, perturbation_type, device=device, batch_size=batch_size)

    return {
        'pred_df': pred_df,
        'auc_df': auc_df,
        'macro_auc': macro_auc,
        'n_images': n_images,
        'n_classes_expected': 14,
        'n_classes_valid': 14,
        'generator_checkpoint_sha256': selected_gen_sha,
        'classifier_checkpoint_sha256': actual_clf_sha,
        'fold': fold,
    }