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
        for i, (inputs, labels, _) in enumerate(dataloader):
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
                idx_name = dataloader.dataset.df.index[batch_size * i + j]
                thisrow = {'Image Index': idx_name}
                truerow = {'Image Index': idx_name}
                for k, lbl in enumerate(PRED_LABEL):
                    thisrow['prob_' + lbl] = probs[j, k]
                    truerow[lbl] = true_labels[j, k]
                pred_df = pd.concat([pred_df, pd.DataFrame(thisrow, index=[0])], ignore_index=True)
                true_df = pd.concat([true_df, pd.DataFrame(truerow, index=[0])], ignore_index=True)

    auc_df = pd.DataFrame(columns=['label', 'auc'])
    for column in true_df.columns:
        if column not in PRED_LABEL:
            continue
        try:
            auc = sklm.roc_auc_score(true_df[column].values.astype(int),
                                     pred_df['prob_' + column].values)
        except Exception:  # noqa: BLE001
            auc = np.nan
        auc_df = pd.concat([auc_df, pd.DataFrame({'label': [column], 'auc': [auc]}, index=[0])],
                           ignore_index=True)

    valid = auc_df['auc'].dropna()
    macro_auc = float(valid.mean()) if len(valid) > 0 else float('nan')
    return pred_df, auc_df, macro_auc


def evaluate_classification_val(config, model=None, fold=DEV_FOLD, device=None,
                                perturbation_type='flow_field', batch_size=16,
                                generator_checkpoint=None):
    """VAL-only classification evaluation. fold is validated BEFORE dataset init.

    :param config: dev config dict (image_path, generator_checkpoint_path, ...).
    :param model: optional injected frozen classifier (tests); defaults to the
        released pretrained_classifier.pth.
    :param fold: development fold; MUST be 'val'.
    :param generator_checkpoint: explicit path to selected M2 generator checkpoint.
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
        actual = file_sha256(ckpt_path)
        if actual != FROZEN_CLASSIFIER_SHA:
            raise RuntimeError('classifier checkpoint SHA drift: %s != frozen %s' % (actual, FROZEN_CLASSIFIER_SHA))

    # Anonymizer (frozen generator, legacy operator, mu=0.01)
    from networks.UNet_PriCheXyNet import UNet

    anonymize_fn = None
    if perturbation_type == 'flow_field':
        gen_path = generator_checkpoint or (config.get('generator_checkpoint_path') if config else None)
        if not gen_path:
            raise ValueError("generator checkpoint path must be explicitly provided")
        generator = UNet(1, 2, 32).to(device)
        generator.load_state_dict(torch.load(gen_path, map_location=device))
        grid_identity, gauss_filter = make_flow_field_components(device)
        anonymize_fn = build_anonymize_fn(generator, grid_identity, gauss_filter, MU)

    # Build VAL-only dataset + loader (fold must be 'val'; never 'test')
    dataset = CXR.CXRDataset(
        path_to_images=config['image_path'],
        fold=fold,
        transform=None,
        perturbation_type=perturbation_type)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    pred_df, auc_df, macro_auc = classify_val_dataset(
        model, dataloader, anonymize_fn, perturbation_type, device=device, batch_size=batch_size)
    return pred_df, auc_df, macro_auc