"""STEP 4A machine summary emitter — Evidence + hashes + verdict (mirrors emit_03h_* pattern)."""

import hashlib
import json
from datetime import datetime, timezone


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    band = sha256('research_agent/band.py')
    triage_src = sha256('research_agent/diag_4a_frozen_triage.py')
    adaptive_src = sha256('research_agent/diag_4a_adaptive_bands.py')
    utility_src = sha256('research_agent/diag_4a_utility_by_band.py')
    spectrum_src = sha256('research_agent/diag_4a_flow_spectrum.py')
    triage = sha256('research_agent/05A_artifacts/frozen_attacker_triage.json')
    adapt = sha256('research_agent/05A_artifacts/adaptive_bands/adaptive_band_results.json')
    utility = sha256('research_agent/05A_artifacts/utility_by_band.json')
    spectrum = sha256('research_agent/05A_artifacts/flow_spectrum.json')
    gen = sha256('networks/corrected_baseline/generator_lowest_total_loss_corrected.pth')

    lp_auc = 0.870531
    hp_auc = 0.865919
    delta = lp_auc - hp_auc
    threshold = 0.05
    verdict = 'FALSIFIED' if delta < 0.05 else ('SUPPORTED' if delta >= 0.10 else 'AMBIGUOUS')

    out = {
        '_label': 'STEP 4A H1 BAND DIAGNOSTIC — machine summary (diagnostic only; TRAIN/VALIDATION; no privacy number)',
        'verdict': verdict,
        'verdict_exact_line': 'STEP 4A H1 DIAGNOSTIC: FALSIFIED',
        'decision': {
            'rule': 'adaptive AUC_lowpass - AUC_highpass; >=0.10 SUPPORTED, 0.05-0.10 AMBIGUOUS, <0.05 FALSIFIED; AUC_lowpass above chance',
            'AUC_lowpass': lp_auc,
            'AUC_highpass': hp_auc,
            'delta': round(delta, 6),
            'falsified_threshold': threshold,
            'auc_lowpass_above_chance': True,
        },
        'adaptive_bands': {
            'seed': 42,
            'restarts_per_band': 1,
            'LP_best_validation_auc': lp_auc,
            'HP_best_validation_auc': hp_auc,
            'endpoint_lowest_validation_loss': True,
            'test_touched': False,
            'reference_unfiltered_mean_auc': 0.8382,
            'reference_unfiltered_sd': 0.0344,
            'reference_seeds': 10,
        },
        'frozen_triage': {
            '_label': 'FROZEN-ATTACKER DISTRIBUTION-SHIFT DIAGNOSTIC (not adaptive; not used in rule)',
            'original_auc': 0.81692,
            'low_pass_auc': 0.70982,
            'high_pass_auc': 0.63589,
        },
        'utility_by_band': {
            '_label': 'VALIDATION-only frozen-model landscape (not part of H1 rule)',
            'classification_mean_auc14': {'original': 0.7938, 'low_pass': 0.6210, 'high_pass': 0.6756},
            'segmentation': {
                'original': {'dice': 0.9550, 'iou': 0.9172, 'hd95_px': 1.307},
                'low_pass': {'dice': 0.8640, 'iou': 0.7680, 'hd95_px': 9.530},
                'high_pass': {'dice': 0.9111, 'iou': 0.8428, 'hd95_px': 5.023},
            },
        },
        'flow_spectrum': {
            'mean_abs_displacement_px': 1.049,
            'max_abs_displacement_px': 1.80,
            'cutoff_cycles_per_px': 0.01656,
            'energy_below_cutoff': 0.108,
            'energy_above_cutoff': 0.892,
        },
        'hashes': {
            'band.py': band,
            'diag_4a_frozen_triage.py': triage_src,
            'diag_4a_adaptive_bands.py': adaptive_src,
            'diag_4a_utility_by_band.py': utility_src,
            'diag_4a_flow_spectrum.py': spectrum_src,
            'frozen_attacker_triage.json': triage,
            'adaptive_band_results.json': adapt,
            'utility_by_band.json': utility,
            'flow_spectrum.json': spectrum,
            'corrected_generator.pth': gen,
        },
        'evaluation_timestamp': datetime.now(timezone.utc).isoformat(),
    }
    with open('research_agent/05A_H1_diagnostic_summary.json', 'w') as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print('verdict =', verdict)
    print('wrote research_agent/05A_H1_diagnostic_summary.json')


if __name__ == '__main__':
    main()