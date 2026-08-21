"""Future role-manifest validator (G0.2 §5).

Operates on synthetic/general patient-ID sets passed in by the caller. Never
reads a real project file and does not decide real split sizes — see
reproduction/reports/G0_1A_FINAL_CLOSEOUT_2026-08-21.md §1.2 for the logical
role definitions this validates against.
"""
import hashlib
import json

ROLES = ('generator_train', 'generator_select', 'attacker_train', 'attacker_select', 'locked_confirm')

# Overlaps that fail closed unconditionally (locked_confirm vs anything, and
# the same-model train/select pairs), even if "whitelisted" by the caller.
_HARD_FORBIDDEN = {
    frozenset({'generator_train', 'generator_select'}),
    frozenset({'attacker_train', 'attacker_select'}),
    frozenset({'locked_confirm', 'generator_train'}),
    frozenset({'locked_confirm', 'generator_select'}),
    frozenset({'locked_confirm', 'attacker_train'}),
    frozenset({'locked_confirm', 'attacker_select'}),
}

# Cross generator/attacker pairs that MAY be allowed if explicitly whitelisted
# with a written justification (role_manifest §5's "predeclared and justified").
_WHITELISTABLE = {
    frozenset({'generator_train', 'attacker_train'}),
    frozenset({'generator_train', 'attacker_select'}),
    frozenset({'generator_select', 'attacker_train'}),
    frozenset({'generator_select', 'attacker_select'}),
}


class RoleManifestError(ValueError):
    pass


def canonical_patient_id(raw) -> str:
    """Canonical string form of a patient identifier: str(), then strip
    surrounding whitespace. Raises on an empty canonical ID."""
    s = str(raw).strip()
    if not s:
        raise RoleManifestError('Empty/whitespace-only patient identifier is not allowed: %r' % (raw,))
    return s


def _check_canonical_collisions(role: str, patients) -> dict:
    """Returns {canonical_id: raw_id} for one role's patient set, raising if
    two DISTINCT raw identifiers canonicalize to the same manifest ID (e.g.
    int 1 and str '1', or ' 1' and '1' — silently distinct as set members,
    but ambiguous once serialized into the manifest)."""
    canon_to_raw = {}
    for raw in patients:
        c = canonical_patient_id(raw)
        if c in canon_to_raw and canon_to_raw[c] != raw:
            raise RoleManifestError(
                "Role %r: distinct patient identifiers %r and %r both canonicalize to %r "
                "— ambiguous manifest ID collision" % (role, canon_to_raw[c], raw, c))
        canon_to_raw[c] = raw
    return canon_to_raw


def validate_role_manifest(roles: dict, whitelist: dict = None):
    """roles: {role_name: set(patient_id)}. whitelist: {frozenset({a,b}): 'justification string'}.

    Raises RoleManifestError (fail-closed) on any unresolved violation.
    Returns True if valid.

    Fix 4 (G0.2A.2): requires set(roles.keys()) == set(ROLES) exactly
    (rejects unexpected/misspelled role names, not just missing ones);
    requires every patient identifier to have a unique, non-empty canonical
    form within its role; requires every whitelist key to be a genuinely
    permitted cross-role pair (not merely checked lazily when an overlap
    happens to occur) with a non-empty justification.
    """
    whitelist = whitelist or {}

    given = set(roles.keys())
    missing = set(ROLES) - given
    unexpected = given - set(ROLES)
    if missing or unexpected:
        raise RoleManifestError(
            'Manifest role set does not exactly match required roles. missing=%s unexpected=%s'
            % (sorted(missing), sorted(unexpected)))

    # Fix (G0.2A.3 Correction 1): overlap must be judged on CANONICAL identities,
    # not raw Python set membership. `generator_train={1}` and
    # `generator_select={"1"}` do not intersect as raw sets (1 != "1" in
    # Python) but both canonicalize to the same manifest ID "1" — a real
    # cross-role overlap that the previous raw-set intersection silently
    # missed. `_check_canonical_collisions` already validates WITHIN-role
    # uniqueness; here we build each role's canonical ID set for the
    # ACROSS-role comparison.
    canonical_sets = {}
    for role, patients in roles.items():
        if not isinstance(patients, (set, frozenset)):
            raise RoleManifestError('Role %r patient set must be a set (got %s)' % (role, type(patients)))
        canon_to_raw = _check_canonical_collisions(role, patients)
        canonical_sets[role] = set(canon_to_raw.keys())

    for pair, justification in whitelist.items():
        if not isinstance(pair, frozenset) or pair not in _WHITELISTABLE:
            raise RoleManifestError('Whitelist key %r is not a permitted cross-role pair' % (pair,))
        if not justification or not str(justification).strip():
            raise RoleManifestError('Whitelist entry for %r has an empty justification' % (pair,))

    names = sorted(roles.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = canonical_sets[a] & canonical_sets[b]
            if not overlap:
                continue
            pair = frozenset({a, b})
            if pair in _HARD_FORBIDDEN:
                raise RoleManifestError(
                    'Forbidden overlap between %r and %r: %d shared canonical patient ID(s) %s'
                    % (a, b, len(overlap), sorted(overlap)))
            if pair in _WHITELISTABLE:
                entry = whitelist.get(pair)
                if not entry or not str(entry).strip():
                    raise RoleManifestError(
                        'Overlap between %r and %r requires an explicit whitelist justification' % (a, b))
                continue
            # Any other overlap combination (roles not in ROLES, or unexpected
            # pairing) fails closed by default.
            raise RoleManifestError(
                'Unrecognized overlap between %r and %r not in the whitelistable set' % (a, b))
    return True


def build_manifest(roles: dict, whitelist: dict = None) -> dict:
    """Validate then produce a deterministic manifest dict (fails closed if invalid).

    Serializes CANONICAL patient IDs (via canonical_patient_id()), and
    `patient_count` is the size of the validated canonical set — not a raw
    len(patients) count, which could disagree from the canonical count in
    principle (though validate_role_manifest already rejects any manifest
    where they would differ, since that implies a within-role collision)."""
    validate_role_manifest(roles, whitelist)
    whitelist = whitelist or {}
    manifest = {
        'roles': {
            role: {
                'patient_count': len({canonical_patient_id(p) for p in patients}),
                'patient_ids_sorted': sorted(canonical_patient_id(p) for p in patients),
            }
            for role, patients in sorted(roles.items())
        },
        'whitelist': {
            '|'.join(sorted(pair)): str(justification)
            for pair, justification in sorted(
                ((tuple(sorted(k)), v) for k, v in whitelist.items()), key=lambda kv: kv[0]
            )
        },
    }
    manifest['manifest_sha256'] = canonical_manifest_hash(manifest)
    return manifest


def canonical_manifest_hash(manifest: dict) -> str:
    """Order-independent SHA256: serializes with sorted keys, excludes any
    pre-existing 'manifest_sha256' field from the hashed payload."""
    payload = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()
