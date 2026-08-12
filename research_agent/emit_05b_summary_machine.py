"""STEP 4B machine summary emitter — Evidence + hashes + verdict (mirrors emit_05a pattern)."""

import hashlib
import json
from datetime import datetime, timezone


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    transforms = sha256('research_agent/caa_transforms.py')
    triage_src = sha256('research_agent/diag_4b_frozen_triage.py')
    adaptive_src = sha256('research_agent/diag_4b_adaptive.py')
    utility_src = sha256('research_agent/diag_4b_utility_sanity.py')
    triage_border = sha256('research_agent/05B_artifacts/frozen_triage_border.json')
    triage_int = sha256('research_agent/05B_artifacts/frozen_triage_intensity.json')
    diag_border = sha256('research_agent/05B_artifacts/adaptive/diag_4b_arm_border_seed42/mechanism_diagnostics.json')
    diag_int = sha256('research_agent/05B_artifacts/adaptive/diag_4b_arm_intensity_seed42/mechanism_diagnostics.json')
    utility = sha256('research_agent/05B_artifacts/utility_sanity.json')

    ref = 0.8382
    h3_auc = 0.79380
    h2_auc = 0.80750
    reduc_h3 = ref - h3_auc
    reduc_h2 = ref - h2_auc
    h3_verdict = 'SUPPORTED' if reduc_h3 >= 0.10 else ('AMBIGUOUS' if reduc_h3 >= 0.05 else 'NOT SUPPORTED')
    h2_verdict = 'SUPPORTED' if reduc_h2 >= 0.10 else ('AMBIGUOUS' if reduc_h2 >= 0.05 else 'NOT SUPPORTED')
    case = 'A' if reduc_h2 >= 0.10 else ('B' if reduc_h3 >= 0.10 else ('C' if (reduc_h2 >= 0.05 or reduc_h3 >= 0.05) else 'D'))
    caa = 'SUPPORTED' if (reduc_h2 >= 0.10 or reduc_h3 >= 0.10) else ('AMBIGUOUS' if (max(reduc_h2, reduc_h3) >= 0.05) else 'FALSIFIED')

    out = {
        '_label': 'STEP 4B CAA MECHANISM DIAGNOSTIC — machine summary (DEVELOPMENT / MECHANISM DIAGNOSTIC; TRAIN/VALIDATION; NOT A PAPER PRIVACY ESTIMATE)',
        'verdict': caa,
        'verdict_exact_line': 'STEP 4B CAA DIAGNOSTIC: ' + caa,
        'case': 'CASE ' + case,
        'reference': {
            'adaptive_reference_mean_auc': ref,
            'adaptive_reference_sd': 0.0344,
            'adaptive_reference_n_seeds': 10,
            'adaptive_reference_source': 'reused 03D/04A compatible VALIDATION per-seed records (no retrain)',
            'generator': 'networks/corrected_baseline/generator_lowest_total_loss_corrected.pth',
            'generator_sha256': sha256('networks/corrected_baseline/generator_lowest_total_loss_corrected.pth'),
            'transform_mode': 'corrected',
            'mu': 0.01,
            'stochastic_lambda': 0.0,
            'pair_train': 'image_pairs/image_pairs_training_10000.txt',
            'pair_validation': 'image_pairs/image_pairs_validation_2000.txt',
        },
        'H3_border': {
            'mechanism': 'border_normalize(BW=4, per-image median fill)',
            'frozen_triage_AUC_reference': 0.81729,
            'frozen_triage_AUC_mechanism': 0.81072,
            'frozen_triage_delta': round(0.81072 - 0.81729, 5),
            'adaptive_seed': 42,
            'adaptive_best_validation_auc': h3_auc,
            'adaptive_reduction': round(reduc_h3, 4),
            'adaptive_verdict': h3_verdict,
            'utility_classification_mean_auc14': 0.7928,
            'utility_segmentation_mean_dice': 0.9528,
            'utility_gross_collapse': False,
            'test_touched': False,
        },
        'H2_intensity': {
            'mechanism': 'intensity_normalize(p1/p99 affine, eps=1e-6)',
            'frozen_triage_AUC_reference': 0.81729,
            'frozen_triage_AUC_mechanism': 0.81699,
            'frozen_triage_delta': round(0.81699 - 0.81729, 5),
            'adaptive_seed': 42,
            'adaptive_best_validation_auc': h2_auc,
            'adaptive_reduction': round(reduc_h2, 4),
            'adaptive_verdict': h2_verdict,
            'utility_classification_mean_auc14': 0.7947,
            'utility_segmentation_mean_dice': 0.9542,
            'utility_gross_collapse': False,
            'test_touched': False,
        },
        'decision': {
            'rule': 'adaptive reduction vs reference; >=0.10 SUPPORTED, 0.05-0.10 AMBIGUOUS, <0.05 NOT SUPPORTED; CAA requires at least one >=0.10 with class>=0.765 and dice>=0.930',
            'H3_reduction': round(reduc_h3, 4),
            'H2_reduction': round(reduc_h2, 4),
            'both_below_005': reduc_h2 < 0.05 and reduc_h3 < 0.05,
        },
        'utility_reference': {
            'classification_mean_auc14': 0.7938,
            'segmentation_mean_dice': 0.9550,
        },
        'interpretation': 'Neither border/FOV nor intensity channel dominates residual identity; with STEP 4A (LP~HP) this supports distributed/redundant identity (H4). CAA not implemented.',
        'hashes': {
            'caa_transforms.py': transforms,
            'diag_4b_frozen_triage.py': triage_src,
            'diag_4b_adaptive.py': adaptive_src,
            'diag_4b_utility_sanity.py': utility_src,
            'frozen_triage_border.json': triage_border,
            'frozen_triage_intensity.json': triage_int,
            'mechanism_diagnostics_border.json': diag_border,
            'mechanism_diagnostics_intensity.json': diag_int,
            'utility_sanity.json': utility,
        },
        'evaluation_timestamp': datetime.now(timezone.utc).isoformat(),
    }
    with open('research_agent/05B_CAA_diagnostic_summary.json', 'w') as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print('verdict =', caa, ' case =', case)
    print('wrote research_agent/05B_CAA_diagnostic_summary.json')


if __name__ == '__main__':
    main()