import os
import json
import argparse

import numpy as np
import pandas as pd
import sklearn.metrics as sklm
import torch
from torchvision import transforms

from networks.UNet_AttentionPriCheXyNet import UNetAtt
from utils.GaussianSmoothing import GaussianSmoothing
import chexnet.cxr_dataset as CXR


_FROZEN_MU = 0.01

PRED_LABEL = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
              'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
              'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def make_pred_multilabel_v2(model, image_path, save_path, perturbation_checkpoint,
                            mu, fold='val', batch_size=16,
                            perturbation_net_class='flow_field_att'):
    """Evaluate a CheXNet classifier on images anonymized by the V2 attention
    generator. Mirrors chexnet.eval_model.make_pred_multilabel's flow_field
    branch step-for-step:
      * dataset yields raw 1-channel [0,1] tensors at 256x256 (Resize+ToTensor),
      * deformation on the RAW image via grid_sample(border, align_corners),
      * THEN expand to 3 channels + Resize(224) + ImageNet normalize,
    with two deliberate differences:
      * perturbation net is UNetAtt with STRICT state loading,
      * the evaluation fold comes from config (default 'val' so development
        stays inside the TRAIN/VAL firewall).
    """

    os.makedirs(save_path, exist_ok=True)
    model.train(False)

    if fold not in ('val', 'test'):
        raise ValueError("fold must be 'val' or 'test'; got %s" % fold)
    # GOVERNANCE (P0.2.2 audit, F2): the official TEST fold is sealed until
    # the paper-final confirmation stage. Any TEST evaluation now requires an
    # EXPLICIT opt-in and is permanently logged.
    if fold == 'test':
        if os.environ.get('ALLOW_TEST_EVAL') != '1':
            raise RuntimeError(
                "fold='test' is governance-locked: the official TEST fold "
                "may only be evaluated at the paper-final confirmation "
                "stage, with ALLOW_TEST_EVAL=1 explicitly set and the run "
                "logged.")
        import time as _time
        with open('./archive/TEST_EVAL_AUDIT_LOG.jsonl', 'a') as _logf:
            _logf.write(json.dumps({
                'event': 'TEST_FOLD_EVALUATION',
                'utc': _time.strftime('%Y-%m-%dT%H:%M:%S+00:00',
                                      _time.gmtime()),
                'mu': mu}) + '\n')
        print('[GOVERNANCE] TEST-fold evaluation authorized via '
              'ALLOW_TEST_EVAL=1; usage logged.')
    if abs(float(mu) - _FROZEN_MU) > 1e-12:
        raise ValueError('mu=%s violates frozen invariant mu=%s.' % (mu, _FROZEN_MU))

    dataset = CXR.CXRDataset(
        path_to_images=image_path,
        fold=fold,
        transform=None,
        # NOTE: the dataset only branches on this string to select its
        # 1-channel raw-image preprocessing path; 'flow_field' selects exactly
        # the path the baseline uses. The generator class is chosen below.
        perturbation_type='flow_field')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False,
                                             num_workers=8)

    # STRICT load: eval graph must equal the trained graph. The class is
    # selected by a separate argument so both V2 (attention) and plain-U-Net
    # control checkpoints can be evaluated through this identical code path.
    if perturbation_net_class == 'flow_field_att':
        perturbation_model = UNetAtt(1, 2, 32).cuda()
    elif perturbation_net_class == 'flow_field':
        from networks.UNet_PriCheXyNet import UNet
        perturbation_model = UNet(1, 2, 32).cuda()
    else:
        raise ValueError("perturbation_net_class must be 'flow_field_att' or "
                         "'flow_field', got '%s'" % perturbation_net_class)
    perturbation_model.load_state_dict(perturbation_checkpoint, strict=True)
    perturbation_model.eval()

    d = torch.linspace(-1, 1, 256)
    mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
    grid_identity = torch.stack((mesh_y, mesh_x), 2)
    grid_identity = grid_identity.unsqueeze(0).permute(0, 3, 1, 2).cuda()
    gauss_filter = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).cuda()

    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pred_rows = []
    true_rows = []

    with torch.no_grad():
        for i, (inputs, labels, indices) in enumerate(dataloader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            grid = perturbation_model(inputs)
            grid = grid_identity - mu * grid
            grid = gauss_filter(grid)
            grid = grid.permute(0, 2, 3, 1)
            inputs = torch.nn.functional.grid_sample(inputs, grid,
                                                     padding_mode='border',
                                                     align_corners=True)

            inputs = trans(inputs.expand(-1, 3, -1, -1))

            probs = model(inputs).cpu().data.numpy()
            true_labels = labels.cpu().data.numpy()

            for j in range(probs.shape[0]):
                idx = indices[j]
                prow = {'Image Index': idx}
                trow = {'Image Index': idx}
                for k, name in enumerate(PRED_LABEL):
                    prow['prob_' + name] = float(probs[j, k])
                    trow[name] = int(true_labels[j, k])
                pred_rows.append(prow)
                true_rows.append(trow)

            if i % 10 == 0:
                print(str(i * batch_size))

    pred_df = pd.DataFrame(pred_rows)
    true_df = pd.DataFrame(true_rows)

    auc_rows = []
    for column in PRED_LABEL:
        actual = true_df[column].values.astype(int)
        pred = pred_df['prob_' + column].values
        auc = np.nan
        try:
            if len(np.unique(actual)) < 2:
                raise ValueError('one-class pathology; ROC-AUC undefined')
            auc = sklm.roc_auc_score(actual, pred)
        except BaseException as e:
            print('cannot calculate auc for %s: %r' % (column, e))
        auc_rows.append({'label': column, 'auc': auc})

    auc_df = pd.DataFrame(auc_rows)
    pred_df.to_csv(save_path + 'preds.csv', index=False)
    auc_df.to_csv(save_path + 'aucs.csv', index=False)

    finite = auc_df['auc'].dropna()
    macro = float(finite.mean())
    summary = {'fold': fold, 'macro_auc_finite_labels': macro,
               'n_finite_labels': int(len(finite)), 'per_label': auc_rows}
    with open(save_path + 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('Macro AUC (%s): %.4f over %d finite labels'
          % (fold, macro, len(finite)))
    return pred_df, auc_df


if __name__ == '__main__':
    print('-------------------------------------------')
    print('-- Evaluate Classifier (PriCheXy-Net V2) --')
    print('-------------------------------------------\n')

    parser = argparse.ArgumentParser('Evaluate Classifier V2')
    parser.add_argument('--config_path', default='./config_files/')
    parser.add_argument('--config', default='config_eval_classifier_v2.json')
    args = parser.parse_args()
    print('Arguments:\n' + '--config_path: ' + args.config_path +
          '\n--config: ' + args.config + '\n')

    # Normalize config path (baseline concatenates verbatim).
    config_path = args.config_path if args.config_path.endswith(os.sep) \
        else args.config_path + os.sep

    with open(config_path + args.config, 'r') as f:
        config = json.loads(f.read())

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # weights_only=False: full pickled module checkpoint (trusted local file).
    checkpoint_best = torch.load(config['classifier_checkpoint'],
                                 map_location='cuda', weights_only=False)
    model = checkpoint_best['model']

    perturbation_checkpoint = None
    if config['perturbation_model_file'] is not None:
        if not os.path.exists(config['perturbation_model_file']):
            raise FileNotFoundError('Perturbation model not found: %s'
                                    % config['perturbation_model_file'])
        perturbation_checkpoint = torch.load(config['perturbation_model_file'],
                                             map_location='cpu')

    make_pred_multilabel_v2(
        model=model.cuda(),
        image_path=config['image_path'],
        save_path=config['save_path'],
        perturbation_checkpoint=perturbation_checkpoint,
        mu=config['mu'],
        fold=config.get('eval_fold', 'val'),
        batch_size=int(config.get('batch_size', 16)),
        perturbation_net_class=config.get('perturbation_type', 'flow_field_att'))
