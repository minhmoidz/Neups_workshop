"""STEP 3B — CONFIRMATORY adaptive Re-ID evaluation of the corrected canonical baseline.

Full pipeline, single immutable arm ``baseline_corrected_confirmatory``:

  STAGE A   train+validate every restart under the frozen confirmatory schedule
            (target 10 VALID, initial seeds 0-9, replacements 10-14 ascending
            issued ONLY for NUMERICALLY_INVALID, max 15 attempts).
  STAGE B   classify execution health (VALID / NUMERICALLY_INVALID) + near-chance.
  STAGE C   select representative attacker from VALIDATION stats only (median best
            validation AUC, tie = lower seed).
  STAGE D   persist/freeze representative + provenance BEFORE any test access.
  STAGE E   official TEST evaluation (frozen image_pairs_testing_5000.txt), exactly
            once per VALID attacker, with a reuse guard (no repeated testing).

All Stages A-D reuse the REAL protocol driver code (run_adaptive_reid_arm.py,
adaptive_reid.*). Stage E is implemented here as a REAL evaluation (the stock
runner's evaluate_test only emits stubs). No test-derived quantity ever enters
Stage C/D. Near-chance runs are retained in every estimand.

Test-set lock during A-D: the official test pair file is not opened until Stage E;
``pair_test_hash`` is withheld until then.
"""

import argparse
import json
import os
import sys

import numpy as np
import sklearn.metrics
import torch

import run_adaptive_reid_arm as R
from adaptive_reid import constants as C
from adaptive_reid import diagnostics as diag
from adaptive_reid import health
from adaptive_reid import pipeline
from adaptive_reid import restarts
from adaptive_reid import summary as summ
from adaptive_reid import topk
from networks.SiameseNetwork import SiameseNetwork
from utils import utils as U
from utils.GaussianSmoothing import GaussianSmoothing

ARM_ID = 'baseline_corrected_confirmatory'
CORRECTED_GENERATOR = 'networks/corrected_baseline/generator_lowest_total_loss_corrected.pth'
DEFAULT_OUT_DIR = './archive/adaptive_reid_baseline_corrected_confirmatory'
PROVENANCE_JSON = 'research_agent/03B_generator_provenance_baseline_corrected.json'


def build_args():
    ap = argparse.ArgumentParser('STEP 3B confirmatory corrected-baseline arm')
    ap.add_argument('--arm_id', default=ARM_ID)
    ap.add_argument('--checkpoint', default=CORRECTED_GENERATOR)
    ap.add_argument('--transform_mode', default='corrected', choices=['legacy', 'corrected'])
    ap.add_argument('--mu', type=float, default=0.01)
    ap.add_argument('--stochastic_lambda', type=float, default=0.0)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--base_config', default=R.CONFIG_TEMPLATE)
    ap.add_argument('--stage', default='a_e', choices=['a_d', 'a_e'])
    ap.add_argument('--mode', default='confirmatory')
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--force', action='store_true')
    return ap.parse_args()


def preflight_frozen_generator(args):
    """Task 1 / STEP 3B header: resolve + verify the frozen generator. STOP on mismatch."""
    prov = diag.read_json(PROVENANCE_JSON)
    path = prov['generator_checkpoint_path']
    sha = prov['generator_checkpoint_sha256']
    actual = diag.sha256_file(path) if os.path.exists(path) else ''
    if actual != sha:
        raise SystemExit('FROZEN GENERATOR MISMATCH: provenance %s != actual %s (path %s)' % (sha, actual, path))
    from adaptive_reid import weights as arw
    if not arw.checkpoint_loadable(path):
        raise SystemExit('FROZEN GENERATOR NOT LOADABLE: %s' % path)
    if prov['transform_mode'] != args.transform_mode:
        raise SystemExit('transform_mode mismatch: provenance %s != args %s'
                         % (prov['transform_mode'], args.transform_mode))
    if float(prov['mu']) != float(args.mu):
        raise SystemExit('mu mismatch: provenance %s != args %s' % (prov['mu'], args.mu))
    if float(prov['stochastic_lambda']) != float(args.stochastic_lambda):
        raise SystemExit('stochastic_lambda mismatch')
    print('PREFLIGHT: frozen generator OK  %s  sha256=%s...' % (path, sha[:12]))
    return prov


def _grid_and_filter():
    d = torch.linspace(-1, 1, 256)
    mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
    grid_identity = torch.stack((mesh_y, mesh_x), 2).unsqueeze(0).permute(0, 3, 1, 2).cuda()
    gauss_filter = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).cuda()
    return grid_identity, gauss_filter


def evaluate_test_real(seed, args, cfg, run_dir):
    """STAGE E — REAL test evaluation for one VALID attacker (Task 7/8).

    D-1: refuses to evaluate a run whose recorded generator digest is stale.
    Task 8: if ``test_metrics.json`` already exists, the recorded result is reused
    (the guard is the file's existence + unchanged generator digest).
    """
    R.verify_stage_e_generator_hash(run_dir, seed, R._generator_hash(args))
    tm_path = os.path.join(run_dir, diag.TESTMETRICS_FILENAME)
    if os.path.exists(tm_path):
        rec = diag.read_json(tm_path)
        if rec.get('generator_checkpoint_hash') == R._generator_hash(args):
            if 'auc' not in rec and 'test_auc' in rec:
                rec = dict(rec)
                rec['auc'] = rec['test_auc']
            return rec
    net = SiameseNetwork().cuda()
    ck = R.attacker_checkpoint_path(seed)
    net.load_state_dict(torch.load(ck, weights_only=False))
    perturbation_net = U.load_flow_generator(args.checkpoint)
    grid_identity, gauss_filter = _grid_and_filter()
    test_loader = U.get_data_loader(phase='testing', experimental_step='retrainSNN',
                                    image_size=int(cfg['image_size']),
                                    n_channels=1, batch_size=int(cfg['batch_size']),
                                    shuffle=False, num_workers=16, pin_memory=True,
                                    image_path=cfg['image_path'])
    y_true, y_pred = U.test_snn('flow_field', net, perturbation_net, grid_identity,
                                gauss_filter, args.mu, test_loader,
                                stochastic_lambda=args.stochastic_lambda,
                                transform_mode=args.transform_mode)
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    auc = float(sklearn.metrics.roc_auc_score(y_true, y_pred))
    test_metrics = {
        'attacker_seed': int(seed),
        'auc': auc,
        'test_auc': auc,
        'n_pairs': int(len(y_true)),
        'valid_for_scientific_reporting': True,
        'stub': False,
        'synthetic': False,
        'generator_checkpoint_path': args.checkpoint,
        'generator_checkpoint_hash': R._generator_hash(args),
        'pair_test_path': R.PAIR_TEST,
        'pair_test_hash': diag.sha256_file(R.PAIR_TEST),
        'transform_mode': args.transform_mode,
        'mu': float(args.mu),
        'stochastic_lambda': float(args.stochastic_lambda),
        'evaluation_timestamp': R.now_iso(),
    }
    diag.write_json(tm_path, test_metrics)
    return test_metrics


def main():
    args = build_args()
    if args.stub:
        raise SystemExit('STEP 3B is a REAL confirmatory arm; --stub not allowed.')
    preflight_frozen_generator(args)

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = R.now_iso()
    with open(args.base_config) as f:
        cfg = json.load(f)

    # --- test-set lock: only train/val hashes during A-D (Task 7/8) -------------
    pair_hashes = R.hashes_for([R.PAIR_TRAIN, R.PAIR_VAL])
    protocol_documents, frozen_artifacts = R.protocol_and_frozen_hashes()
    commit = R.git_commit()

    def train_and_report(seed):
        exp = 'retrain_snn_seed{}'.format(seed)
        run_dir = os.path.join(args.out_dir, 'runs', exp)
        os.makedirs(run_dir, exist_ok=True)
        start = R.now_iso()
        signature = R.run_signature(seed, args, cfg, pair_hashes,
                                    protocol_documents, frozen_artifacts)
        diag.write_json(os.path.join(run_dir, R.SIGNATURE_FILENAME), signature)
        reused = None
        if not args.force:
            reused = R.reuse_completed_run(run_dir, signature, seed, args)
        if reused is not None:
            rec = reused
        else:
            rec = R._train_once(seed, exp, run_dir, start, args, cfg, pair_hashes,
                                protocol_documents, frozen_artifacts)
        rec['run_end_timestamp'] = R.now_iso()
        diag.write_json(os.path.join(run_dir, diag.VALIDITY_FILENAME), rec)
        state, near = health.classify_run_health(rec)
        validation_record = R._validation_record_from(rec)
        diag.write_json(os.path.join(run_dir, diag.RUNSTATE_FILENAME),
                        {'state': state, 'near_chance': near, 'evaluated_test': False})
        return {'attacker_seed': seed, 'state': state, 'near_chance': near,
                'diagnostics': rec, 'validation_record': validation_record,
                'run_dir': run_dir}

    # ---- STAGES A-D: frozen confirmatory schedule (target 10, replacements only
    #      for NUMERICALLY_INVALID, ascending 10-14, max 15 attempts). -----------
    schedule = restarts.ConfirmatorySchedule()
    attempts = restarts.run_schedule(schedule, train_and_report)

    pl = pipeline.ArmPipeline(train_validate_and_persist=lambda seed: train_and_report(seed),
                              evaluate_test=None)
    pl.stage_b_classify(attempts)
    rep_seed = pl.stage_c_select_representative(attempts)
    mark = {}
    pl.stage_d_persist_representative(attempts, rep_seed, mark)

    summary = summ.summarize_arm(attempts)
    summary['representative_attacker_seed'] = rep_seed

    t_stage_d = R.now_iso()
    prov = R.write_arm_provenance(args.arm_id, args.out_dir, summary, attempts, args, cfg,
                                  pair_hashes, t0, t_stage_d, commit)
    prov['pair_test_hash'] = 'WITHHELD_TEST_SET_LOCK'
    prov['representative_selection_timestamp'] = t_stage_d
    diag.write_json(os.path.join(args.out_dir, 'arm_provenance_stageD.json'), prov)
    diag.write_json(os.path.join(args.out_dir, 'arm_summary_stageD.json'), summary)
    print('STAGE D FROZEN: representative_attacker_seed =', rep_seed)
    print(json.dumps(summary, indent=2, sort_keys=True))

    # ---- STAGE E (only with --stage a_e): official TEST, exactly once/valid. ----
    if args.stage == 'a_e':
        for a in attempts:
            if a['state'] != C.VALID:
                a['test_metrics'] = None
                continue
            a['test_metrics'] = evaluate_test_real(a['attacker_seed'], args, cfg, a['run_dir'])
            runstate = diag.read_json(os.path.join(a['run_dir'], diag.RUNSTATE_FILENAME))
            runstate['evaluated_test'] = True
            diag.write_json(os.path.join(a['run_dir'], diag.RUNSTATE_FILENAME), runstate)
        summary = summ.summarize_arm(attempts)
        summary['representative_attacker_seed'] = rep_seed
        t1 = R.now_iso()
        prov = R.write_arm_provenance(args.arm_id, args.out_dir, summary, attempts, args, cfg,
                                      pair_hashes, t0, t1, commit)
        prov['pair_test_hash'] = diag.sha256_file(R.PAIR_TEST)
        prov['representative_selection_timestamp'] = t_stage_d
        diag.write_json(os.path.join(args.out_dir, 'arm_provenance.json'), prov)
        diag.write_json(os.path.join(args.out_dir, 'arm_summary.json'), summary)

    frozen = topk.load_frozen_topk_list_canonical(R.FROZEN_TOPK_CSV,
                                                  expected_sha256=topk.FROZEN_TOPK_SHA256)
    topk.save_frozen_topk_list(os.path.join(args.out_dir, topk.FROZEN_LIST_FILENAME), frozen)

    print('ARM COMPLETE: %s' % args.arm_id)
    print('attempts:', len(attempts))
    print('valid:', sum(1 for a in attempts if a['state'] == C.VALID))
    print('invalid:', sum(1 for a in attempts if a['state'] == C.NUMERICALLY_INVALID))
    print('near_chance:', sum(1 for a in attempts if a['near_chance']))
    print('representative_attacker_seed:', rep_seed)
    print('scientific_summary_available:', summary.get('scientific_summary_available'))
    print('test AUCs:', summary.get('test_auc_values'))
    print('mean/std/median/max:', summary.get('mean_test_auc'), summary.get('std_test_auc'),
          summary.get('median_test_auc'), summary.get('max_test_auc'))


if __name__ == '__main__':
    sys.exit(main())
