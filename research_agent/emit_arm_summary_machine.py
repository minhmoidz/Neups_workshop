"""STEP 3B.1 T6 — emit arm_summary_machine.json through the audited summarize_arm().

Consumes ONLY persisted per-seed records (run_state.json, training_diagnostics.json,
test_metrics.json) and the Stage-D provenance, all byte-for-byte copies under
research_agent/03D_artifacts/. No inference, no test images, no recomputation of
predictions. The test AUC is taken verbatim from the recorded test_metrics.json.

The canonical audited Stage-E reuse path (run_3b_confirmatory.py:evaluate_test_real)
reads a persisted test_metrics.json and, when it carries ``test_auc`` but not ``auc``,
adds the ``auc`` alias before feeding records to summarize_arm (run_3b lines 108-112).
This script applies exactly that adapter so summarize_arm's documented contract
('test_metrics carries the auc key') is satisfied without modifying any record.
"""

import json
import os

from adaptive_reid import summary as summ

BUNDLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '03D_artifacts')

OUT_PATH = os.path.join(BUNDLE, 'arm_summary_machine.json')


def read_json(p):
    with open(p) as f:
        return json.load(f)


def main():
    prov = read_json(os.path.join(BUNDLE, 'arm_provenance.json'))
    rep_seed = int(prov['representative_attacker_seed'])
    seeds = [int(s) for s in prov['attacker_seeds_attempted']]

    attempts = []
    for seed in seeds:
        base = os.path.join(BUNDLE, 'seed_%d' % seed)
        run_state = read_json(os.path.join(base, 'run_state.json'))
        diag_rec = read_json(os.path.join(base, 'training_diagnostics.json'))
        test_metrics = read_json(os.path.join(base, 'test_metrics.json'))
        # Canonical adapter (run_3b_confirmatory.py evaluate_test_real reuse path):
        if 'auc' not in test_metrics and 'test_auc' in test_metrics:
            test_metrics = dict(test_metrics)
            test_metrics['auc'] = test_metrics['test_auc']
        attempts.append({
            'attacker_seed': seed,
            'state': run_state['state'],
            'near_chance': bool(run_state['near_chance']),
            'test_metrics': test_metrics,
            'is_representative': (seed == rep_seed),
            'run_dir': os.path.join(base, 'runs', 'retrain_snn_seed%d' % seed),
        })

    # Audited aggregator: computes mean/SD(ddof=1)/median/max and the scientific
    # availability / stub-synthetic firewall itself.
    summary = summ.summarize_arm(attempts)
    summary['representative_attacker_seed'] = rep_seed

    # Record which artifacts fed the summary (pointers only, no recomputation).
    summary['source_bundle'] = 'research_agent/03D_artifacts/'
    summary['source_arm_id'] = prov['arm_id']
    summary['source_commit'] = prov['git_commit']

    with open(OUT_PATH, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print('WROTE', OUT_PATH)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
