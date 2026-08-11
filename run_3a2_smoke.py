"""STEP 3A2 — ONE real adaptive attacker smoke (seed 0) against the frozen
corrected generator. Stages A-D ONLY. No Stage E, no test split access.

Reuses the REAL protocol driver code (run_adaptive_reid_arm.py, adaptive_reid.*):
- exactly ONE training invocation (seed 0), no replacement seed
- idempotent reuse on a second invocation (no --force)
- real health classification, real pipeline Stage B/C/D, real provenance
- D-1 content pinning: generator path + SHA-256 bound into run signature

Test-set lock: the official test pair file is NEVER opened. pair_hashes contains
only the train and validation pair files; pair_test_hash is intentionally withheld.
"""

import argparse
import json
import os
import sys

import run_adaptive_reid_arm as R
from adaptive_reid import diagnostics as diag
from adaptive_reid import health
from adaptive_reid import pipeline
from adaptive_reid import summary as summ
from adaptive_reid import topk

SMOKE_ARM_ID = 'arm_smoke_3a2_corrected_seed0'
CORRECTED_GENERATOR = 'networks/corrected_baseline/generator_lowest_total_loss_corrected.pth'
DEFAULT_OUT_DIR = './archive/adaptive_reid_smoke_3a2'


def build_args():
    ap = argparse.ArgumentParser('STEP 3A2 single-seed attacker smoke (A-D only)')
    ap.add_argument('--arm_id', default=SMOKE_ARM_ID)
    ap.add_argument('--checkpoint', default=CORRECTED_GENERATOR)
    ap.add_argument('--transform_mode', default='corrected', choices=['legacy', 'corrected'])
    ap.add_argument('--mu', type=float, default=0.01)
    ap.add_argument('--stochastic_lambda', type=float, default=0.0)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--base_config', default=R.CONFIG_TEMPLATE)
    ap.add_argument('--stage', default='a_d', choices=['a_d', 'a_e'])
    ap.add_argument('--mode', default='smoke_single_seed')
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--count_file', default=None,
                    help='path to a counter file; a line is appended for EACH real '
                         'training invocation (used to prove reuse == 0 invocations)')
    return ap.parse_args()


def main():
    args = build_args()
    if args.stage != 'a_d':
        raise SystemExit('STEP 3A2 smoke runs stages A-D ONLY (--stage a_d). No Stage E.')
    if args.stub:
        raise SystemExit('STEP 3A2 smoke is a REAL run; --stub is not allowed.')

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = R.now_iso()

    with open(args.base_config) as f:
        cfg = json.load(f)

    # Test-set lock: hashes for the TRAIN and VALIDATION pair files only.
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
            if args.count_file:
                with open(args.count_file, 'a') as f:
                    f.write('{}\n'.format(seed))

        rec['run_end_timestamp'] = R.now_iso()
        diag.write_json(os.path.join(run_dir, diag.VALIDITY_FILENAME), rec)

        state, near = health.classify_run_health(rec)
        validation_record = R._validation_record_from(rec)
        diag.write_json(os.path.join(run_dir, diag.RUNSTATE_FILENAME),
                        {'state': state, 'near_chance': near, 'evaluated_test': False})
        return {'attacker_seed': seed, 'state': state, 'near_chance': near,
                'diagnostics': rec, 'validation_record': validation_record,
                'run_dir': run_dir, 'reused': reused is not None,
                'training_invoked': reused is None}

    # ---- STAGE A: exactly ONE attempt, seed 0 only. No replacement. ----
    attempts = [train_and_report(0)]

    # ---- STAGE B / C / D: real pipeline, representative selection is canonical. ----
    pl = pipeline.ArmPipeline(
        train_validate_and_persist=lambda seed: train_and_report(seed),
        evaluate_test=None)  # Stage E disabled
    pl.stage_b_classify(attempts)
    rep_seed = pl.stage_c_select_representative(attempts)
    mark = {}
    pl.stage_d_persist_representative(attempts, rep_seed, mark)

    summary = summ.summarize_arm(attempts)
    summary['representative_attacker_seed'] = rep_seed

    t1 = R.now_iso()
    prov = R.write_arm_provenance(args.arm_id, args.out_dir, summary, attempts, args, cfg,
                                  pair_hashes, t0, t1, commit)
    # Test-set lock note: pair_test_hash is withheld (never opened).
    prov['pair_test_hash'] = 'WITHHELD_TEST_SET_LOCK'
    diag.write_json(os.path.join(args.out_dir, 'arm_provenance.json'), prov)
    diag.write_json(os.path.join(args.out_dir, 'arm_summary.json'), summary)

    frozen = topk.load_frozen_topk_list_canonical(R.FROZEN_TOPK_CSV,
                                                  expected_sha256=topk.FROZEN_TOPK_SHA256)
    topk.save_frozen_topk_list(os.path.join(args.out_dir, topk.FROZEN_LIST_FILENAME), frozen)

    print('ARM COMPLETE: %s' % args.arm_id)
    print('attempts:', len(attempts))
    print('seed0 reused (no training):', attempts[0]['reused'])
    print('training invoked this invocation:', attempts[0]['training_invoked'])
    print('representative_attacker_seed:', rep_seed)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print('provenance:', os.path.join(args.out_dir, 'arm_provenance.json'))


if __name__ == '__main__':
    sys.exit(main())
