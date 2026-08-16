"""T201–T216: M1.4c.2 Final Closeout Hotfix Test Suite.

Closes the classification replay integrity + audit hygiene gaps:

  - T201  production classifier writer schema is <label> / prob_<label>
  - T202  valid production replay finds exactly 14 pathologies -> VALID
  - T203  missing ground-truth column invalidates run
  - T204  missing probability column invalidates run
  - T205  13-row AUC CSV invalidates run
  - T206  duplicate AUC pathology invalidates run
  - T207  unknown AUC pathology invalidates run
  - T208  per-class AUC mismatch invalidates run
  - T209  macro AUC mismatch invalidates run
  - T210  production writer -> CSV -> replay exact PASS (end-to-end)
  - T211  production artifact missing prob_Hernia -> INVALID
  - T212  privacy replay tamper detected
  - T213  batch-size report wording == 16 pairs
  - T214  SD sample convention verified (ddof=1)
  - T215  promotion fileset contains no forbidden historical runtime baggage
  - T216  protocol authority hierarchy is unambiguous

All replay tests exercise the REAL production writer schema and the REAL
serializer -> file -> check_run_validity() loop.
"""
import copy
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_AGENT_DIR = os.path.dirname(TESTS_DIR)
ROOT = os.path.dirname(RESEARCH_AGENT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if RESEARCH_AGENT_DIR not in sys.path:
    sys.path.insert(0, RESEARCH_AGENT_DIR)

from m2_dev.evaluator_common import (
    file_sha256,
    NIH_PATHOLOGIES,
    REQUIRED_PATHOLOGY_COLUMNS,
)
from m2_dev import run_m2_s1
from m2_dev.eval_classifier_val import evaluate_classification_val
from m2_dev.eval_classifier_val import classify_val_dataset


# ---------------------------------------------------------------------------
# Helpers — build production-schema classification artifacts the same way the
# production writer (classify_val_dataset + evaluate_classification_arm) does.
# ---------------------------------------------------------------------------
def _make_production_pred_df(n_rows=4, seed=7):
    """Construct a pred_df in the EXACT structure returned by classify_val_dataset()."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_rows):
        row = {'Image Index': 'img_%04d.png' % i}
        for p in NIH_PATHOLOGIES:
            row[p] = int(rng.randint(0, 2))
            row['prob_' + p] = float(rng.rand())
        rows.append(row)
    return pd.DataFrame(rows)


def _serialize_production_files(pred_df, auc_df, tmp, pred_name='class_pred.csv', auc_name='class_aucs.csv'):
    """Serialize with the SAME pandas calls production evaluate_classification_arm uses."""
    pred_p = os.path.join(tmp, pred_name)
    auc_p = os.path.join(tmp, auc_name)
    pred_df.to_csv(pred_p, index=False)
    auc_df.to_csv(auc_p, index=False)
    return pred_p, file_sha256(pred_p), auc_p, file_sha256(auc_p)


def _auc_df_from_pred(pred_df):
    """Compute the production auc_df (label/auc) from a pred_df using production AUC math."""
    import sklearn.metrics as sklm
    auc_rows = []
    for p in NIH_PATHOLOGIES:
        auc = float(sklm.roc_auc_score(pred_df[p].values.astype(int), pred_df['prob_' + p].values.astype(float)))
        auc_rows.append({'label': p, 'auc': auc})
    return pd.DataFrame(auc_rows)


def _base_manifests(tmp, gen_p, gen_sha, att_p, att_sha):
    b_dev_m = {'epochs_completed': 250, 'requested_max_epochs': 250, 'numerical_validity': 'PASS', 'nan_inf_detected': False,
               'selected_generator_checkpoint': gen_p, 'selected_generator_sha256': gen_sha}
    c4_m = copy.deepcopy(b_dev_m)
    b_att_m = {'best_attacker_path': att_p, 'best_attacker_sha256': att_sha, 'generator_checkpoint_sha256': gen_sha,
               'numerical_validity': 'PASS', 'nan_inf_detected': False}
    c4_att_m = copy.deepcopy(b_att_m)
    return b_dev_m, c4_m, b_att_m, c4_att_m


def _base_privacy(tmp, gen_sha, att_sha, y_true=None, y_score=None):
    y_true = np.array([0, 1, 0, 1]) if y_true is None else y_true
    y_score = np.array([0.1, 0.9, 0.2, 0.8]) if y_score is None else y_score
    npz_p = os.path.join(tmp, 'privacy.npz')
    np.savez_compressed(npz_p, y_true=y_true, y_score=y_score)
    npz_sha = file_sha256(npz_p)
    b_priv = {'roc_auc': 1.0, 'generator_checkpoint_sha256': gen_sha, 'attacker_checkpoint_sha256': att_sha, 'n_pairs': 4,
              'predictions_file': npz_p, 'predictions_file_sha256': npz_sha}
    c4_priv = copy.deepcopy(b_priv)
    return b_priv, c4_priv


def _valid_check(tmp, b_class, c4_class, gen_p=None, gen_sha=None, att_p=None, att_sha=None,
                 b_priv=None, c4_priv=None, expected_epochs=250):
    """Run check_run_validity on a full bundle with defaults that pass."""
    if gen_p is None:
        gen_p = os.path.join(tmp, 'gen.pth')
        torch.save({}, gen_p)
    gen_sha = gen_sha or file_sha256(gen_p)
    if att_p is None:
        att_p = os.path.join(tmp, 'att.pth')
        torch.save({}, att_p)
    att_sha = att_sha or file_sha256(att_p)
    b_dev_m, c4_m, b_att_m, c4_att_m = _base_manifests(tmp, gen_p, gen_sha, att_p, att_sha)
    if b_priv is None:
        b_priv, c4_priv = _base_privacy(tmp, gen_sha, att_sha)
    # Propagate the real generator SHA into the classification results so the
    # generator SHA link contract is satisfied (production does this via
    # evaluate_classification_arm -> evaluate_classification_val).
    for c_res in (b_class, c4_class):
        c_res['generator_checkpoint_sha256'] = gen_sha
    return run_m2_s1.check_run_validity(
        b_dev_m, c4_m, b_att_m, c4_att_m, b_priv, c4_priv, b_class, c4_class,
        expected_epochs=expected_epochs, unit_test_mode=True)


def _production_class_bundle(tmp, pred_df, auc_df, macro_auc):
    pred_p, pred_sha, auc_p, auc_sha = _serialize_production_files(pred_df, auc_df, tmp)
    b_class = {'macro_auc': macro_auc, 'n_classes_valid': 14, 'generator_checkpoint_sha256': 'G',
               'n_images': len(pred_df),
               'predictions_file': pred_p, 'predictions_file_sha256': pred_sha,
               'aucs_file': auc_p, 'aucs_file_sha256': auc_sha, 'auc_df': auc_df}
    c4_class = copy.deepcopy(b_class)
    return b_class, c4_class


# ---------------------------------------------------------------------------
# T201: production writer schema
# ---------------------------------------------------------------------------
def test_t201_production_classification_writer_schema():
    """T201: classify_val_dataset output uses <label> and prob_<label> columns."""
    with tempfile.TemporaryDirectory() as tmp:
        pred_df, auc_df = None, None
        try:
            # Build a tiny injected classifier that outputs 14-dim probs so we
            # exercise the REAL classify_val_dataset() writer loop on synthetic data.
            import torch.nn as nn
            class TinyClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.pool = nn.AdaptiveAvgPool2d((1, 1))
                    self.fc = nn.Linear(3, 14)

                def forward(self, x):
                    return torch.sigmoid(self.fc(self.pool(x).flatten(1)))

            from m0_tests.test_m14a_execution_harness import SyntheticClassificationDataset
            ds = SyntheticClassificationDataset(size=28, image_size=64)
            loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
            model = TinyClassifier()
            pred_df, auc_df, macro_auc = classify_val_dataset(
                model, loader, anonymize_fn=None, perturbation_type='none', device='cpu', batch_size=8)
        except Exception as e:  # pragma: no cover - defensive
            raise AssertionError("classify_val_dataset failed to run: %s" % e)

        for p in NIH_PATHOLOGIES:
            assert p in pred_df.columns, "Missing ground-truth column %r" % p
            assert ('prob_' + p) in pred_df.columns, "Missing probability column %r" % ('prob_' + p)
        assert 'Image Index' in pred_df.columns
        assert list(auc_df['label']) == NIH_PATHOLOGIES
        assert list(auc_df.columns) == ['label', 'auc']
        assert len(auc_df) == 14
    return True


# ---------------------------------------------------------------------------
# T202: valid production replay -> VALID
# ---------------------------------------------------------------------------
def test_t202_valid_production_replay_finds_14():
    """T202: correct production artifacts replay -> VALID with exactly 14 pathologies."""
    with tempfile.TemporaryDirectory() as tmp:
        pred_df = _make_production_pred_df(n_rows=64, seed=11)
        auc_df = _auc_df_from_pred(pred_df)
        macro_auc = float(auc_df['auc'].mean())
        b_class, c4_class = _production_class_bundle(tmp, pred_df, auc_df, macro_auc)
        valid, msg = _valid_check(tmp, b_class, c4_class)
        assert valid is True, "Expected VALID, got: %s" % msg
    return True


# ---------------------------------------------------------------------------
# R1: missing one GT pathology column
# ---------------------------------------------------------------------------
def test_t203_missing_gt_column_invalidates():
    """T203: raw prediction CSV missing one GT pathology column -> INVALID."""
    with tempfile.TemporaryDirectory() as tmp:
        pred_df = _make_production_pred_df(n_rows=64, seed=12)
        pred_df = pred_df.drop(columns=['Hernia'])
        auc_df = _auc_df_from_pred(_make_production_pred_df(n_rows=64, seed=12))
        macro_auc = float(auc_df['auc'].mean())
        b_class, c4_class = _production_class_bundle(tmp, pred_df, auc_df, macro_auc)
        valid, msg = _valid_check(tmp, b_class, c4_class)
        assert valid is False
        assert 'missing ground-truth column' in msg and 'Hernia' in msg
    return True


# ---------------------------------------------------------------------------
# R2: missing one probability column
# ---------------------------------------------------------------------------
def test_t204_missing_probability_column_invalidates():
    """T204: raw prediction CSV missing one probability column -> INVALID."""
    with tempfile.TemporaryDirectory() as tmp:
        pred_df = _make_production_pred_df(n_rows=64, seed=13)
        pred_df = pred_df.drop(columns=['prob_Hernia'])
        auc_df = _auc_df_from_pred(_make_production_pred_df(n_rows=64, seed=13))
        macro_auc = float(auc_df['auc'].mean())
        b_class, c4_class = _production_class_bundle(tmp, pred_df, auc_df, macro_auc)
        valid, msg = _valid_check(tmp, b_class, c4_class)
        assert valid is False
        assert 'missing probability column' in msg and 'prob_Hernia' in msg
    return True


# ---------------------------------------------------------------------------
# R3: AUC CSV contains only 13 pathologies
# ---------------------------------------------------------------------------
def test_t205_thirteen_row_auc_csv_invalidates():
    """T205: AUC CSV with only 13 pathologies -> INVALID."""
    with tempfile.TemporaryDirectory() as tmp:
        pred_df = _make_production_pred_df(n_rows=64, seed=14)
        auc_df = _auc_df_from_pred(pred_df)
        auc_df_13 = auc_df[auc_df['label'] != 'Hernia']
        macro_auc = float(auc_df['auc'].mean())
        b_class, c4_class = _production_class_bundle(tmp, pred_df, auc_df_13, macro_auc)
        valid, msg = _valid_check(tmp, b_class, c4_class)
        assert valid is False
        assert '14 rows' in msg
    return True


# ---------------------------------------------------------------------------
# R4: AUC CSV duplicate pathology
# ---------------------------------------------------------------------------
def test_t206_duplicate_auc_pathology_invalidates():
    """T206: AUC CSV with a duplicate pathology row -> INVALID."""
    with tempfile.TemporaryDirectory() as tmp:
        pred_df = _make_production_pred_df(n_rows=64, seed=15)
        auc_df = _auc_df_from_pred(pred_df)
        # Exactly 14 rows but with a duplicate (Hernia repeated) and one missing
        # pathology (Edema removed) so the row count is 14 yet duplicates exist.
        dup = auc_df[auc_df['label'] != 'Edema']
        dup = pd.concat([dup, auc_df[auc_df['label'] == 'Hernia']], ignore_index=True)
        assert len(dup) == 14 and dup['label'].nunique() == 13
        macro_auc = float(auc_df['auc'].mean())
        b_class, c4_class = _production_class_bundle(tmp, pred_df, dup, macro_auc)
        valid, msg = _valid_check(tmp, b_class, c4_class)
        assert valid is False
        assert 'duplicate pathology' in msg
    return True


# ---------------------------------------------------------------------------
# T207: unknown AUC pathology
# ---------------------------------------------------------------------------
def test_t207_unknown_auc_pathology_invalidates():
    """T207: AUC CSV with an unknown pathology row -> INVALID."""
    with tempfile.TemporaryDirectory() as tmp:
        pred_df = _make_production_pred_df(n_rows=64, seed=16)
        auc_df = _auc_df_from_pred(pred_df)
        bad = auc_df.copy()
        bad.loc[0, 'label'] = 'NotAPathology'
        macro_auc = float(auc_df['auc'].mean())
        b_class, c4_class = _production_class_bundle(tmp, pred_df, bad, macro_auc)
        valid, msg = _valid_check(tmp, b_class, c4_class)
        assert valid is False
        assert 'unknown/missing pathologies' in msg
    return True


# ---------------------------------------------------------------------------
# R7: reported per-pathology AUC altered
# ---------------------------------------------------------------------------
def test_t208_per_class_auc_mismatch_invalidates():
    """T208: reported per-pathology AUC (auc_df) altered -> INVALID."""
    with tempfile.TemporaryDirectory() as tmp:
        pred_df = _make_production_pred_df(n_rows=64, seed=17)
        auc_df = _auc_df_from_pred(pred_df)
        macro_auc = float(auc_df['auc'].mean())
        bad_auc_df = auc_df.copy()
        bad_auc_df.loc[0, 'auc'] = 1.0 if bad_auc_df.loc[0, 'auc'] < 0.99 else 0.5
        b_class, c4_class = _production_class_bundle(tmp, pred_df, auc_df, macro_auc)
        # Overwrite in-memory auc_df with tampered one
        b_class['auc_df'] = bad_auc_df
        c4_class['auc_df'] = bad_auc_df.copy()
        valid, msg = _valid_check(tmp, b_class, c4_class)
        assert valid is False
        assert 'per-pathology replay mismatch' in msg
    return True


# ---------------------------------------------------------------------------
# R6: reported macro AUC altered
# ---------------------------------------------------------------------------
def test_t209_macro_auc_mismatch_invalidates():
    """T209: reported macro AUC altered -> INVALID."""
    with tempfile.TemporaryDirectory() as tmp:
        pred_df = _make_production_pred_df(n_rows=64, seed=18)
        auc_df = _auc_df_from_pred(pred_df)
        macro_auc = float(auc_df['auc'].mean())
        b_class, c4_class = _production_class_bundle(tmp, pred_df, auc_df, macro_auc)
        b_class['macro_auc'] = 0.123
        c4_class['macro_auc'] = 0.123
        valid, msg = _valid_check(tmp, b_class, c4_class)
        assert valid is False
        assert 'replayed macro AUC mismatch' in msg
    return True


# ---------------------------------------------------------------------------
# T210: production writer -> CSV -> replay exact PASS (end-to-end)
# ---------------------------------------------------------------------------
def test_t210_production_writer_to_replay_exact_pass():
    """T210: evaluate_classification_arm writes real production files; replay -> VALID."""
    with tempfile.TemporaryDirectory() as tmp:
        from networks.UNet_PriCheXyNet import UNet

        # Two distinct generator checkpoints so both arms have valid SHA links.
        gen_b = os.path.join(tmp, 'B_dev', 'seed_42', 'gen_b.pth')
        gen_c = os.path.join(tmp, 'C4', 'seed_42', 'gen_c.pth')
        os.makedirs(os.path.dirname(gen_b), exist_ok=True)
        os.makedirs(os.path.dirname(gen_c), exist_ok=True)
        torch.save(UNet(1, 2, 32).state_dict(), gen_b)
        torch.save(UNet(1, 2, 32).state_dict(), gen_c)
        gen_b_sha = file_sha256(gen_b)
        gen_c_sha = file_sha256(gen_c)

        for arm, gen_p, gen_sha in [('B_dev', gen_b, gen_b_sha), ('C4', gen_c, gen_c_sha)]:
            manifest = {'selected_generator_checkpoint': gen_p, 'selected_generator_sha256': gen_sha}
            with open(os.path.join(tmp, arm, 'seed_42', 'checkpoint_manifest.json'), 'w') as f:
                json.dump(manifest, f)

        b_clf = run_m2_s1.evaluate_classification_arm('B_dev', 42, 'cpu', out_base_dir=tmp, unit_test_mode=True)
        c4_clf = run_m2_s1.evaluate_classification_arm('C4', 42, 'cpu', out_base_dir=tmp, unit_test_mode=True)

        # The production serializer already wrote classification_val_predictions.csv
        # and classification_val_aucs.csv into each arm dir.
        assert os.path.exists(b_clf['predictions_file'])
        assert os.path.exists(b_clf['aucs_file'])
        assert os.path.exists(c4_clf['predictions_file'])
        assert os.path.exists(c4_clf['aucs_file'])

        b_priv, c4_priv = _base_privacy(tmp, gen_b_sha, 'D', y_true=np.array([0, 1, 0, 1]),
                                        y_score=np.array([0.1, 0.9, 0.2, 0.8]))
        c4_priv['generator_checkpoint_sha256'] = gen_c_sha
        b_att = {'best_attacker_path': os.path.join(tmp, 'att_b.pth'), 'best_attacker_sha256': 'D',
                 'generator_checkpoint_sha256': gen_b_sha, 'numerical_validity': 'PASS', 'nan_inf_detected': False}
        c4_att = copy.deepcopy(b_att)
        c4_att['generator_checkpoint_sha256'] = gen_c_sha
        torch.save({}, b_att['best_attacker_path'])
        torch.save({}, c4_att['best_attacker_path'])
        b_att['best_attacker_sha256'] = file_sha256(b_att['best_attacker_path'])
        c4_att['best_attacker_sha256'] = file_sha256(c4_att['best_attacker_path'])
        b_priv['attacker_checkpoint_sha256'] = b_att['best_attacker_sha256']
        c4_priv['attacker_checkpoint_sha256'] = c4_att['best_attacker_sha256']

        b_dev_m = {'epochs_completed': 250, 'requested_max_epochs': 250, 'numerical_validity': 'PASS', 'nan_inf_detected': False,
                   'selected_generator_checkpoint': gen_b, 'selected_generator_sha256': gen_b_sha}
        c4_m = {'epochs_completed': 250, 'requested_max_epochs': 250, 'numerical_validity': 'PASS', 'nan_inf_detected': False,
                'selected_generator_checkpoint': gen_c, 'selected_generator_sha256': gen_c_sha}

        valid, msg = run_m2_s1.check_run_validity(
            b_dev_m, c4_m, b_att, c4_att, b_priv, c4_priv, b_clf, c4_clf,
            expected_epochs=250, unit_test_mode=True)
        assert valid is True, "End-to-end production replay must be VALID: %s" % msg
    return True


# ---------------------------------------------------------------------------
# T211: production artifact missing prob_Hernia -> INVALID
# ---------------------------------------------------------------------------
def test_t211_production_artifact_missing_prob_hernia_invalidates():
    """T211: end-to-end production files, then remove prob_Hernia -> INVALID."""
    with tempfile.TemporaryDirectory() as tmp:
        from networks.UNet_PriCheXyNet import UNet

        gen_b = os.path.join(tmp, 'B_dev', 'seed_42', 'gen_b.pth')
        os.makedirs(os.path.dirname(gen_b), exist_ok=True)
        torch.save(UNet(1, 2, 32).state_dict(), gen_b)
        gen_b_sha = file_sha256(gen_b)
        with open(os.path.join(tmp, 'B_dev', 'seed_42', 'checkpoint_manifest.json'), 'w') as f:
            json.dump({'selected_generator_checkpoint': gen_b, 'selected_generator_sha256': gen_b_sha}, f)

        b_clf = run_m2_s1.evaluate_classification_arm('B_dev', 42, 'cpu', out_base_dir=tmp, unit_test_mode=True)

        # Mutate ONE production column: remove prob_Hernia; update SHA metadata.
        pred_p = b_clf['predictions_file']
        pred_df = pd.read_csv(pred_p)
        assert 'prob_Hernia' in pred_df.columns
        pred_df.drop(columns=['prob_Hernia']).to_csv(pred_p, index=False)
        b_clf['predictions_file_sha256'] = file_sha256(pred_p)

        c4_clf = copy.deepcopy(b_clf)
        b_priv, c4_priv = _base_privacy(tmp, gen_b_sha, 'D', y_true=np.array([0, 1, 0, 1]),
                                        y_score=np.array([0.1, 0.9, 0.2, 0.8]))
        b_att = {'best_attacker_path': os.path.join(tmp, 'att.pth'), 'best_attacker_sha256': 'D',
                 'generator_checkpoint_sha256': gen_b_sha, 'numerical_validity': 'PASS', 'nan_inf_detected': False}
        c4_att = copy.deepcopy(b_att)
        torch.save({}, b_att['best_attacker_path'])
        torch.save({}, c4_att['best_attacker_path'])
        b_att['best_attacker_sha256'] = file_sha256(b_att['best_attacker_path'])
        c4_att['best_attacker_sha256'] = file_sha256(c4_att['best_attacker_path'])
        b_priv['attacker_checkpoint_sha256'] = b_att['best_attacker_sha256']
        c4_priv['attacker_checkpoint_sha256'] = c4_att['best_attacker_sha256']

        b_dev_m = {'epochs_completed': 250, 'requested_max_epochs': 250, 'numerical_validity': 'PASS', 'nan_inf_detected': False,
                   'selected_generator_checkpoint': gen_b, 'selected_generator_sha256': gen_b_sha}
        c4_m = copy.deepcopy(b_dev_m)

        valid, msg = run_m2_s1.check_run_validity(
            b_dev_m, c4_m, b_att, c4_att, b_priv, c4_priv, b_clf, c4_clf,
            expected_epochs=250, unit_test_mode=True)
        assert valid is False, "Removed prob_Hernia must invalidate the run"
        assert 'missing probability column' in msg and 'prob_Hernia' in msg
    return True


# ---------------------------------------------------------------------------
# T212: privacy replay tamper
# ---------------------------------------------------------------------------
def test_t212_privacy_replay_tamper_detected():
    """T212: tampering y_score in privacy NPZ (with updated SHA) -> INVALID."""
    with tempfile.TemporaryDirectory() as tmp:
        gen_p = os.path.join(tmp, 'gen.pth')
        att_p = os.path.join(tmp, 'att.pth')
        torch.save({}, gen_p)
        torch.save({}, att_p)
        gen_sha = file_sha256(gen_p)
        att_sha = file_sha256(att_p)

        npz_p = os.path.join(tmp, 'privacy.npz')
        np.savez_compressed(npz_p, y_true=np.array([0, 1, 0, 1]), y_score=np.array([0.1, 0.9, 0.2, 0.8]))

        pred_df = _make_production_pred_df(n_rows=64, seed=20)
        auc_df = _auc_df_from_pred(pred_df)
        macro_auc = float(auc_df['auc'].mean())
        b_class, c4_class = _production_class_bundle(tmp, pred_df, auc_df, macro_auc)

        # Tamper y_score so the replayed AUC differs (1.0 -> 0.0); update SHA metadata.
        np.savez_compressed(npz_p, y_true=np.array([0, 1, 0, 1]), y_score=np.array([0.8, 0.1, 0.9, 0.2]))
        tampered_sha = file_sha256(npz_p)
        b_priv = {'roc_auc': 1.0, 'generator_checkpoint_sha256': gen_sha, 'attacker_checkpoint_sha256': att_sha, 'n_pairs': 4,
                  'predictions_file': npz_p, 'predictions_file_sha256': tampered_sha}
        c4_priv = copy.deepcopy(b_priv)

        b_dev_m = {'epochs_completed': 250, 'requested_max_epochs': 250, 'numerical_validity': 'PASS', 'nan_inf_detected': False,
                   'selected_generator_checkpoint': gen_p, 'selected_generator_sha256': gen_sha}
        c4_m = copy.deepcopy(b_dev_m)
        b_att_m = {'best_attacker_path': att_p, 'best_attacker_sha256': att_sha, 'generator_checkpoint_sha256': gen_sha,
                   'numerical_validity': 'PASS', 'nan_inf_detected': False}
        c4_att_m = copy.deepcopy(b_att_m)

        valid, msg = run_m2_s1.check_run_validity(
            b_dev_m, c4_m, b_att_m, c4_att_m, b_priv, c4_priv, b_class, c4_class,
            expected_epochs=250, unit_test_mode=True)
        assert valid is False
        assert 'privacy replayed AUC mismatch' in msg
    return True


# ---------------------------------------------------------------------------
# T213: batch-size report wording
# ---------------------------------------------------------------------------
def test_t213_batch_size_report_wording_16_pairs():
    """T213: M1.4c report must describe batch_size=16 as 16 pair samples (not 8 pairs)."""
    rep_p = os.path.join(RESEARCH_AGENT_DIR, 'M1_4C_FINAL_FORENSIC_CERTIFICATION.md')
    with open(rep_p) as f:
        text = f.read()
    assert '16 images (`8` pairs)' not in text, "Report still contains incorrect '8 pairs' wording"
    assert '16 pair samples' in text and '16 pairs' in text
    return True


# ---------------------------------------------------------------------------
# T214: SD sample convention
# ---------------------------------------------------------------------------
def test_t214_sd_sample_convention_verified():
    """T214: upstream 10-run JSON must expose clearly-labelled ddof=0 and ddof=1 SD."""
    res_p = os.path.join(RESEARCH_AGENT_DIR, 'upstream_10run_reproduction_results.json')
    with open(res_p) as f:
        data = json.load(f)
    block = data['upstream_10_runs_retrained_attacker_legacy']
    assert 'std_auc_ddof0' in block, "Missing labelled ddof=0 population SD field"
    assert 'std_auc_sample_ddof1' in block, "Missing labelled ddof=1 sample SD field"
    assert abs(block['std_auc_ddof0'] - 0.028215762629424015) < 1e-12
    assert abs(block['std_auc_sample_ddof1'] - 0.02974202527588046) < 1e-12
    return True


# ---------------------------------------------------------------------------
# T215: promotion fileset
# ---------------------------------------------------------------------------
def test_t215_promotion_fileset_no_forbidden_baggage():
    """T215: M1_4C2_PROMOTION_FILESET.json must exist and classify no runtime baggage."""
    fs_p = os.path.join(RESEARCH_AGENT_DIR, 'M1_4C2_PROMOTION_FILESET.json')
    assert os.path.exists(fs_p), "Promotion fileset missing: %s" % fs_p
    with open(fs_p) as f:
        data = json.load(f)
    include = data.get('include_in_canonical_promotion', [])
    retain = data.get('retain_on_audit_branch_only', [])
    include_basenames = [os.path.basename(p) for p in include]
    retain_basenames = [os.path.basename(p) for p in retain]
    forbidden = [
        '20_UPSTREAM_EXACT_REPRODUCTION_AUDIT_summary.json',
        'audit_operator_equivalence.py',
        'audit_splits_and_pairs.py',
        'checkpoint_inventory.json',
        'environment_comparison.json',
        'reproduce_upstream_val_metrics.py',
        'run_complete_upstream_reproduction.py',
        'upstream_10run_reproduction_results.json',
        'upstream_compatibility.patch',
        'upstream_validation_classification_utility.json',
    ]
    for fname in forbidden:
        assert fname not in include_basenames, "Forbidden historical runtime baggage listed for promotion: %s" % fname
        assert fname in retain_basenames, "Forbidden file %s not explicitly retained on audit branch only" % fname
    # Required production files must be in the promotion include set.
    for required in ['research_agent/m2_dev/run_m2_s1.py', 'research_agent/m2_dev/eval_classifier_val.py',
                     'research_agent/m2_dev/evaluator_common.py']:
        assert any(required in inc for inc in include), "Required production file %s missing from promotion set" % required
    return True


# ---------------------------------------------------------------------------
# T216: protocol authority hierarchy
# ---------------------------------------------------------------------------
def test_t216_protocol_authority_hierarchy_unambiguous():
    """T216: PROTOCOL_AUTHORITY.md must assign distinct roles to execution lock vs certification manifest."""
    pa_p = os.path.join(RESEARCH_AGENT_DIR, 'PROTOCOL_AUTHORITY.md')
    with open(pa_p) as f:
        text = f.read()
    assert 'M2_S1_EXECUTION_LOCK.json' in text
    assert 'M1_4C_CERTIFICATION_MANIFEST.json' in text
    assert 'scientific method / frozen scientific execution choices' in text or 'scientific method' in text
    assert 'certification evidence' in text
    assert 'sole authoritative' in text
    assert 'must not override' in text or 'must NOT' in text or 'Do NOT' in text
    return True


def run_all():
    tests = [
        test_t201_production_classification_writer_schema,
        test_t202_valid_production_replay_finds_14,
        test_t203_missing_gt_column_invalidates,
        test_t204_missing_probability_column_invalidates,
        test_t205_thirteen_row_auc_csv_invalidates,
        test_t206_duplicate_auc_pathology_invalidates,
        test_t207_unknown_auc_pathology_invalidates,
        test_t208_per_class_auc_mismatch_invalidates,
        test_t209_macro_auc_mismatch_invalidates,
        test_t210_production_writer_to_replay_exact_pass,
        test_t211_production_artifact_missing_prob_hernia_invalidates,
        test_t212_privacy_replay_tamper_detected,
        test_t213_batch_size_report_wording_16_pairs,
        test_t214_sd_sample_convention_verified,
        test_t215_promotion_fileset_no_forbidden_baggage,
        test_t216_protocol_authority_hierarchy_unambiguous,
    ]
    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            res = t()
            if res:
                print(f"  [PASS] {name}")
                passed += 1
            else:
                print(f"  [FAIL] {name}")
                failed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\nM1.4c.2 Closeout Suite: {passed}/{len(tests)} PASS, {failed} FAIL")
    return failed == 0


if __name__ == '__main__':
    ok = run_all()
    sys.exit(0 if ok else 1)