"""Patient-clustered bootstrap (STEP 2B Part 9, R-9).

Pair-file schema (verified by inspection of image_pairs/*):
    each line:  <image1> <image2> <label>
    label 1 (positive): both images carry the SAME patient identity.
    label 0 (negative): the two images carry TWO DIFFERENT patient identities.

Patient-cluster (identity) bootstrap resamples at the level of patient identities and
re-weights/among resampled clusters. A positive pair belongs unambiguously to exactly
one patient cluster. A negative pair spans TWO clusters, so its membership under a
cluster resample is ambiguous: the frozen protocol does not specify how such a pair is
assigned when patient clusters are resampled.

Per the frozen protocol, in a case of scientific ambiguity we DO NOT guess: R-9 is
reported BLOCKED FOR SCIENTIFIC CLARIFICATION, and this module documents the exact
ambiguity. The historical pair-level bootstrap is retained only as a
pair-sampling diagnostic (clearly labelled).
"""

from typing import Dict, List, Tuple


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
    returns True for the frozen pair files.
    """
    stats = compute_patient_statistics(pairs_path)
    return stats['negative_pairs_span_two_patients'] > 0


def report_R9_ambiguity(pairs_path: str) -> Dict:
    """Produce the R-9 BLOCKED documentation record.

    :return: dict describing the exact ambiguity, so it can be persisted with the arm.
    """
    stats = compute_patient_statistics(pairs_path)
    return {
        'R9_status': 'BLOCKED_FOR_SCIENTIFIC_CLARIFICATION',
        'ambiguity': (
            'Negative pairs span two patient identities; the frozen protocol does not '
            'define how such a pair is assigned when patient clusters are resampled. '
            'No clustering rule invented here.'),
        'pair_statistics': stats,
        'pair_bootstrap_retained_as': 'pair_sampling_diagnostic',
    }


class PatientClusterResampler:
    """Explicit patient-cluster resampling core.

    The cluster membership rule for a pair is defined explicitly by the caller through
    ``pair_to_clusters``. This class does NOT invent a rule; for the frozen pair files
    it will refuse to resample unless the caller resolves the negative-pair ambiguity.
    """

    def __init__(self, pair_to_clusters):
        """
        :param pair_to_clusters: callable (image1, image2, label) -> tuple of patient
            cluster ids that the pair belongs to. For a positive pair this is a single
            id; the caller decides the rule for negative pairs.
        """
        self.pair_to_clusters = pair_to_clusters

    def resample(self, rows, rng, n_clusters_to_draw):
        """Resample a patient-cluster-level bootstrap sample of pair rows.

        :param rows: list of (image1, image2, label) rows.
        :param rng: numpy RandomState.
        :param n_clusters_to_draw: number of patient clusters to draw with replacement.
        :return: list of pairs whose clusters intersect the drawn cluster set, scaled
            as a diagnostic count (resource-efficient: returns indices).
        """
        cluster_ids = set()
        for row in rows:
            cluster_ids.update(self.pair_to_clusters(*row))
        cluster_ids = sorted(cluster_ids)
        drawn = {c for c in rng.choice(cluster_ids, size=n_clusters_to_draw, replace=True)}
        kept = [i for i, row in enumerate(rows) if drawn & set(self.pair_to_clusters(*row))]
        weights = [len(drawn) for _ in kept]  # diagnostic weighting placeholder
        return kept, weights