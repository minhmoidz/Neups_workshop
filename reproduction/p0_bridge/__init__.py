"""P0 reproduction-only bridge infrastructure (CPU-tested design).

Revision P0_2_1 — external source-review closeout:
- explicit domain-separated seed contract with per-arm seed bundles;
- deterministic epoch sampler independent of global RNG (P0_SAMPLER_V1_1);
- validated order hashes (P0_ORDERHASH_V1_1);
- hardened frozen-generator state guard and model-state hashing
  (P0_MODELSTATE_V1_1);
- protocol-aware fail-closed manifest aggregation with fresh-output claims;
- hardened execution approval gate with actual artifact byte verification.

Reproduction-only: no scientific loop is implemented; nothing here grants
execution authorization. Protocol schema: P0_PROTOCOL_V1_1.
"""
