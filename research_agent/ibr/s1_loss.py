"""Phase-II IBR S1 - loss assembly (single place the S1 objective is computed).

Assembles the five-term S1 loss for one batch, using the frozen STEP 6A
coefficients. This is the exact graph the future training entry point uses.

Loss terms:
    L_rec   : L1(x_self, x)                          -- ONLY on x_self
    L_path  : BCE(classifier(x_anon), y_path_source)
    L_anat  : MSE(seg(x_anon), seg(x))               -- source anatomy target
    L_zid   : BCE(verifier(z_id, z_id_donor), y_pair)
    L_adv   : BCE(adv(GRL(z_med), GRL(z_med_donor)), y_pair)

NOTE on pairwise identity for the z_id objective: the STEP 6A lock defines the
pair (x1, x2) with identity label y_id. In the S1 batch we form the pair from
the source and the donor image. y_pair = 1 iff donor and source share a patient
(this is False by the donor protocol's hard invariant, but the objective is
defined generally so the same code serves positive/negative pairs).
"""

import torch

from research_agent.ibr.losses import (
    LAMBDA_REC, LAMBDA_PATH, LAMBDA_ANAT, LAMBDA_ZID, LAMBDA_ADV,
    reconstruction_loss, classification_loss, anatomy_loss,
    zid_pair_loss, zmed_adv_loss,
)


def compute_s1_loss(model, frozen_utility, x, x_donor, y_path, x_pair, y_pair, return_parts=False):
    """Compute the full S1 loss for a batch.

    Args:
        model: IBRModel
        frozen_utility: FrozenUtility (frozen classifier + segmentation teacher)
        x: source images (B,1,256,256) [-1,1]
        x_donor: donor images (B,1,256,256) [-1,1]; used ONLY for the anon
            branch (x_anon = G(z_id_donor, z_med_source)). The donor contributes
            identity via z_id; the donor patient is always != source patient.
        y_path: (B,14) source pathology labels in {0,1}
        x_pair: (B,1,256,256) partner images forming the (source, partner)
            identity pair. Partner may be same-patient (y_pair=1) or
            different-patient (y_pair=0); both classes MUST be present.
        y_pair: (B,1) same/different-patient labels for the (x, x_pair) identity
            pairs. This is the STEP 6A lock §3/§4 pair label y_id.
        return_parts: if True, also return the dict of individual terms

    Returns:
        total loss (scalar tensor), or (total, parts dict) when return_parts=True.
    """
    out = model(x, x_donor)
    x_self, x_anon = out['x_self'], out['x_anon']
    z_id, z_med = out['z_id'], out['z_med']

    L_rec = reconstruction_loss(x_self, x)

    # x_anon constrained by task/anatomy/privacy terms only (never pixel rec).
    path_prob = frozen_utility.path_logits(x_anon)
    L_path = classification_loss(path_prob, y_path)
    anat_anon = frozen_utility.anat_maps(x_anon)
    anat_source = frozen_utility.anat_maps(x)
    L_anat = anatomy_loss(anat_anon, anat_source)

    # identity pair (source, partner) with BOTH classes in y_pair (lock §3/§4).
    z_id_pair, z_med_pair, _ = model.encode(x_pair)
    L_zid = zid_pair_loss(model.verify(z_id, z_id_pair), y_pair)
    L_adv = zmed_adv_loss(model.adversary_logits(z_med, z_med_pair), y_pair)

    total = (LAMBDA_REC * L_rec + LAMBDA_PATH * L_path + LAMBDA_ANAT * L_anat
             + LAMBDA_ZID * L_zid + LAMBDA_ADV * L_adv)

    if return_parts:
        parts = {'L_rec': L_rec.item(), 'L_path': L_path.item(), 'L_anat': L_anat.item(),
                 'L_zid': L_zid.item(), 'L_adv': L_adv.item(),
                 'total': total.item()}
        return total, parts
    return total