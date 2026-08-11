"""Uncertainty / bootstrap policy (STEP 2B Part 9, R-9 — FINAL).

Pair-file schema (verified by inspection of image_pairs/*):
    each line:  <image1> <image2> <label>
    label 1 (positive): both images carry the SAME patient identity.
    label 0 (negative): the two images carry TWO DIFFERENT patient identities.

The verification data are therefore DYADIC: a negative pair has no unique one-way
patient-cluster membership, so a naive patient-cluster bootstrap would duplicate or
ambiguously assign negative-pair observations.

Final ruling (01B amendment §6, protocol §12.2 / R-9):
    - the patient-clustered bootstrap proposal is WITHDRAWN;
    - primary uncertainty for privacy claims is the distribution over independently
      trained attacker restarts (mean, sample SD ddof=1, unpaired Welch);
    - the existing PAIR-LEVEL bootstrap may remain only as a clearly labelled secondary
      diagnostic for the validation-selected representative attacker:

          PAIR-SAMPLING DIAGNOSTIC — NOT PATIENT-LEVEL UNCERTAINTY

    - NO patient-cluster resampling implementation exists in this module. If a pair
      bootstrap is ever used it must carry the label above.
"""

from typing import Dict, List, Tuple

PAIR_BOOTSTRAP_LABEL = 'PAIR-SAMPLING DIAGNOSTIC — NOT PATIENT-LEVEL UNCERTAINTY'
R9_FINAL_STATUS = 'WITHDRAWN_PATIENT_CLUSTER_BOOTSTRAP'


def parse_patient_ids(fname_pair_row) -> Tuple[str, str]:
    """Patient identity = the numeric prefix before the first '_' in an NIH file name."""
    id1 = fname_pair_row[0].split('_')[0]
    id2 = fname_pair_row[1].split('_')[0]
    return id1, id2


def load_pairs(path: str) -> List[Tuple[str, str, float]]:
    """Load a pair file into (image1, image2, label) rows."""
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError('malformed pair row: %r' % line)
            rows.append((parts[0], parts[1], float(parts[2])))
    return rows


def compute_patient_statistics(pairs_path: str):
    """Return per-file statistics used to characterise the pair schema.

    :return: dict with positive_pairs_single_patient, negative_pairs_span_two_patients,
        and similar counts.
    """
    rows = load_pairs(pairs_path)
    positive = [r for r in rows if r[2] == 1.0]
    negative = [r for r in rows if r[2] == 0.0]

    pos_single = sum(1 for r in positive
                     if parse_patient_ids(r)[0] == parse_patient_ids(r)[1])
    neg_two = sum(1 for r in negative
                  if parse_patient_ids(r)[0] != parse_patient_ids(r)[1])

    return {
        'n_pairs': len(rows),
        'n_positive': len(positive),
        'n_negative': len(negative),
        'positive_pairs_single_patient': pos_single,
        'negative_pairs_span_two_patients': neg_two,
        'any_positive_pair_spans_two_patients': pos_single != len(positive),
        'any_negative_pair_within_one_patient': neg_two != len(negative),
    }


def patient_cluster_bootstrap_is_ambiguous(pairs_path: str) -> bool:
    """Return True iff the pair schema makes patient-cluster bootstrap ambiguous.

    The schema may be un-ambiguous ONLY if every pair involves at most one patient
    identity. Our verified schema has negative pairs spanning two identities, so this
    returns True for the frozen pair files. This is the reason the estimator was
    withdrawn rather than guessed (R-9 / amendment §6).
    """
    stats = compute_patient_statistics(pairs_path)
    return stats['negative_pairs_span_two_patients'] > 0


def report_R9_final_policy(pairs_path: str) -> Dict:
    """Produce the R-9 FINAL policy documentation record.

    :return: dict describing why the patient-cluster bootstrap is withdrawn and how any
        pair-level bootstrap must be labelled, so it can be persisted with the arm.
    """
    stats = compute_patient_statistics(pairs_path)
    return {
        'R9_status': R9_FINAL_STATUS,
        'withdrawn_because': (
            'Verification pairs are dyadic: negative pairs span two patient identities, '
            'so a naive patient-cluster bootstrap would duplicate or ambiguously assign '
            'negative-pair observations and must not be described as patient-level '
            'uncertainty (01B amendment §6; protocol §12.2).'),
        'pair_statistics': stats,
        'primary_uncertainty': 'distribution over independently trained attacker restarts',
        'restart_sd_ddof': 1,
        'pair_bootstrap_label': PAIR_BOOTSTRAP_LABEL,
    }
