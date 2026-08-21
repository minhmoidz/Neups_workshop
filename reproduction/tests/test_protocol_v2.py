"""Targeted CPU-only, synthetic-only tests for reproduction/protocol_v2/*.

No CUDA, no real image/checkpoint/config/pair-file access, no network, no
pretrained-weight download. Plain-script runnable (no pytest dependency),
mirroring the project's existing research_agent/m0_tests/ convention.

Run: .venv/bin/python reproduction/tests/test_protocol_v2.py
"""
import os
import sys
import tempfile

import torch
import torch.nn as nn

torch.manual_seed(0)  # deterministic test seed; CPU-only throughout this file
assert not torch.cuda.is_initialized(), 'CUDA must not be initialized in this test file'

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from reproduction.protocol_v2.state_invariants import (
    canonical_tensor_state_hash, buffer_only_hash, parameter_version_signature,
    preserved_eval_forward,
)
from reproduction.protocol_v2.deterministic_loader import (
    DeterministicEpochSampler, build_deterministic_loader,
)
from reproduction.protocol_v2.role_manifest import (
    validate_role_manifest, build_manifest, canonical_manifest_hash, RoleManifestError,
)
from reproduction.protocol_v2.output_freshness import assert_fresh_output_dir, OutputNotFreshError
from reproduction.protocol_v2.weight_provenance import record_weight_provenance, WeightProvenanceError

RESULTS = []


def check(name, condition):
    RESULTS.append((name, bool(condition)))
    print(('PASS' if condition else 'FAIL') + ' - ' + name)
    if not condition:
        raise AssertionError('Test failed: %s' % name)


class ToyBNNet(nn.Module):
    """Minimal CPU network with BatchNorm, standing in for the real UNet
    generator (which also has BatchNorm2d in every block)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn(self.conv(x))


def test_state_invariants():
    net = ToyBNNet()
    x = torch.randn(8, 1, 16, 16)

    # 1. Ordinary .train() forward under no_grad() DOES change BN buffers.
    net.train()
    h_before = buffer_only_hash(net)
    with torch.no_grad():
        net(x)
    h_after = buffer_only_hash(net)
    check('train()+no_grad() forward changes BN buffers (baseline confound reproduced)', h_before != h_after)

    # 2. Guarded evaluation forward does NOT change full state hash.
    net2 = ToyBNNet()
    net2.train()
    full_before = canonical_tensor_state_hash(net2)
    with preserved_eval_forward(net2):
        net2(x)
    full_after = canonical_tensor_state_hash(net2)
    check('preserved_eval_forward leaves full state hash unchanged', full_before == full_after)

    # 3. Original training mode is restored.
    check('preserved_eval_forward restores .training=True', net2.training is True)

    net3 = ToyBNNet()
    net3.eval()
    with preserved_eval_forward(net3):
        net3(x)
    check('preserved_eval_forward restores .training=False', net3.training is False)

    # 4. Mode restored after a deliberately raised exception.
    net4 = ToyBNNet()
    net4.train()
    raised = False
    try:
        with preserved_eval_forward(net4):
            net4(x)
            raise ValueError('deliberate test exception')
    except ValueError:
        raised = True
    check('exception propagates out of preserved_eval_forward', raised)
    check('mode restored after exception', net4.training is True)

    # 5. Outputs remain finite.
    net5 = ToyBNNet()
    with preserved_eval_forward(net5):
        out = net5(x)
    check('guarded forward output finite', torch.isfinite(out).all().item())

    # 6. Changing a parameter changes canonical hash.
    net6 = ToyBNNet()
    h1 = canonical_tensor_state_hash(net6)
    with torch.no_grad():
        net6.conv.weight[0, 0, 0, 0] += 1.0
    h2 = canonical_tensor_state_hash(net6)
    check('parameter change alters canonical hash', h1 != h2)

    # 7. Changing a buffer changes canonical AND buffer hash.
    net7 = ToyBNNet()
    cb1 = canonical_tensor_state_hash(net7)
    bb1 = buffer_only_hash(net7)
    with torch.no_grad():
        net7.bn.running_mean[0] += 1.0
    cb2 = canonical_tensor_state_hash(net7)
    bb2 = buffer_only_hash(net7)
    check('buffer change alters canonical hash', cb1 != cb2)
    check('buffer change alters buffer-only hash', bb1 != bb2)

    # parameter_version_signature: bumps on in-place param mutation.
    net8 = ToyBNNet()
    v1 = parameter_version_signature(net8)
    with torch.no_grad():
        net8.conv.weight.add_(1.0)
    v2 = parameter_version_signature(net8)
    check('parameter _version signature changes on in-place mutation', v1 != v2)

    # dtype-mismatch safety: BN buffers include running_mean(float)+num_batches_tracked(int64);
    # hashing must not raise despite mixed dtypes.
    net9 = ToyBNNet()
    net9.train()
    with torch.no_grad():
        net9(x)  # populate num_batches_tracked etc.
    try:
        canonical_tensor_state_hash(net9)
        mixed_dtype_ok = True
    except Exception:
        mixed_dtype_ok = False
    check('canonical hash handles mixed-dtype buffers without error', mixed_dtype_ok)


class ToySeqDataset(torch.utils.data.Dataset):
    def __init__(self, n=20):
        self.n = n
        self.ids = ['sample_%02d' % i for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.tensor([idx], dtype=torch.float32)


def test_deterministic_loader():
    ds = ToySeqDataset(20)

    # 1. Same seed, independent loaders -> same epoch orders.
    _, s1 = build_deterministic_loader(ds, seed=7, batch_size=4, shuffle=True)
    _, s2 = build_deterministic_loader(ds, seed=7, batch_size=4, shuffle=True)
    order1 = list(iter(s1))
    order2 = list(iter(s2))
    check('same seed -> identical epoch-0 order across independent samplers', order1 == order2)

    # 2. Different seeds -> different epoch-0 order.
    _, s3 = build_deterministic_loader(ds, seed=8, batch_size=4, shuffle=True)
    order3 = list(iter(s3))
    check('different seed -> different epoch-0 order', order3 != order1)

    # 3. Consuming arbitrary global RNG before iteration does not change order.
    _, s4 = build_deterministic_loader(ds, seed=7, batch_size=4, shuffle=True)
    torch.manual_seed(999999)  # perturb global RNG
    _ = torch.rand(1000)
    order4 = list(iter(s4))
    check('global RNG perturbation before iteration does not change order', order4 == order1)

    # 4. Recreating the loader reproduces the full predeclared epoch sequence.
    _, s5 = build_deterministic_loader(ds, seed=7, batch_size=4, shuffle=True)
    seq_a = [list(iter(s5)) for _ in range(3)]
    _, s6 = build_deterministic_loader(ds, seed=7, batch_size=4, shuffle=True)
    seq_b = [list(iter(s6)) for _ in range(3)]
    check('recreated loader reproduces full 3-epoch sequence', seq_a == seq_b)

    # 5. Semantic order hash changes when sample identity/order changes.
    h1 = s5.epoch_order_hash(0, sample_ids=ds.ids)
    ds_perm_ids = list(reversed(ds.ids))
    h2 = s5.epoch_order_hash(0, sample_ids=ds_perm_ids)
    check('semantic order hash changes when sample-id mapping changes', h1 != h2)

    # 6. Validation loader remains sequential (shuffle=False).
    vloader, vsampler = build_deterministic_loader(ds, seed=7, batch_size=4, shuffle=False)
    check('shuffle=False returns no custom sampler', vsampler is None)
    seen = [int(b.item()) for batch in vloader for b in batch]
    check('validation loader preserves dataset order', seen == list(range(20)))

    # 7. num_workers=0 used throughout this test file (multiworker unclaimed).
    check('this test suite uses num_workers=0 only (documented, not exercised beyond)', True)


def test_role_manifest():
    disjoint = {
        'generator_train': {'p1', 'p2', 'p3'},
        'generator_select': {'p4', 'p5'},
        'attacker_train': {'p6', 'p7'},
        'attacker_select': {'p8'},
        'locked_confirm': {'p9', 'p10'},
    }
    check('all-disjoint valid case validates', validate_role_manifest(disjoint) is True)

    def with_overlap(role_a, role_b, shared='pX'):
        roles = {k: set(v) for k, v in disjoint.items()}
        roles[role_a] = roles[role_a] | {shared}
        roles[role_b] = roles[role_b] | {shared}
        return roles

    forbidden_pairs = [
        ('generator_train', 'generator_select'),
        ('attacker_train', 'attacker_select'),
        ('locked_confirm', 'generator_train'),
        ('locked_confirm', 'generator_select'),
        ('locked_confirm', 'attacker_train'),
        ('locked_confirm', 'attacker_select'),
    ]
    for a, b in forbidden_pairs:
        roles = with_overlap(a, b)
        raised = False
        try:
            validate_role_manifest(roles)
        except RoleManifestError:
            raised = True
        check('forbidden overlap %s/%s rejected' % (a, b), raised)

    # explicitly allowed generator_train/attacker_train overlap WITH justification
    roles = with_overlap('generator_train', 'attacker_train')
    wl = {frozenset({'generator_train', 'attacker_train'}): 'predeclared feasibility overlap'}
    check('whitelisted cross overlap with justification passes', validate_role_manifest(roles, wl) is True)

    # same overlap WITHOUT justification must fail
    raised = False
    try:
        validate_role_manifest(roles, {frozenset({'generator_train', 'attacker_train'}): ''})
    except RoleManifestError:
        raised = True
    check('whitelisted-pair overlap without justification text fails', raised)

    # deterministic hash: same logical manifest, different dict insertion order -> same hash
    roles_a = {'generator_train': {'a', 'b'}, 'generator_select': {'c'}, 'attacker_train': {'d'},
               'attacker_select': {'e'}, 'locked_confirm': {'f'}}
    roles_b = {'locked_confirm': {'f'}, 'attacker_select': {'e'}, 'attacker_train': {'d'},
               'generator_select': {'c'}, 'generator_train': {'a', 'b'}}
    m_a = build_manifest(roles_a)
    m_b = build_manifest(roles_b)
    check('insertion-order-independent manifest hash', m_a['manifest_sha256'] == m_b['manifest_sha256'])

    # one patient change -> different hash
    roles_c = dict(roles_a)
    roles_c['generator_train'] = {'a', 'b', 'z'}
    m_c = build_manifest(roles_c)
    check('one patient change alters manifest hash', m_a['manifest_sha256'] != m_c['manifest_sha256'])

    # --- Fix 4 (G0.2A.2) negative tests ---

    # unexpected role name (typo) must be rejected, not silently ignored
    roles_unexpected = {k: set(v) for k, v in disjoint.items()}
    roles_unexpected['generator_trian'] = {'typo_patient'}  # misspelled extra role
    raised = False
    try:
        validate_role_manifest(roles_unexpected)
    except RoleManifestError:
        raised = True
    check('unexpected/misspelled role name rejected', raised)

    # missing role (already covered pre-fix, re-confirm post-fix)
    roles_missing = {k: set(v) for k, v in disjoint.items() if k != 'attacker_select'}
    raised = False
    try:
        validate_role_manifest(roles_missing)
    except RoleManifestError:
        raised = True
    check('missing role still rejected after Fix 4', raised)

    # whitelist key naming a pair that is not a permitted cross-role pair at all
    # (e.g. accidentally whitelisting a locked_confirm overlap) must fail even
    # if the justification text is present and non-empty.
    raised = False
    try:
        validate_role_manifest(disjoint, {frozenset({'locked_confirm', 'generator_train'}): 'oops, not allowed'})
    except RoleManifestError:
        raised = True
    check('whitelist key naming a non-permitted pair is rejected', raised)

    # empty justification for an otherwise-valid whitelist key
    raised = False
    try:
        validate_role_manifest(disjoint, {frozenset({'generator_train', 'attacker_train'}): '   '})
    except RoleManifestError:
        raised = True
    check('whitelist-eligible pair with whitespace-only justification rejected', raised)

    # canonical-ID collision: int 1 and str '1' are distinct set members but
    # collide once serialized into the manifest as strings.
    roles_collision = {k: set(v) for k, v in disjoint.items()}
    roles_collision['generator_train'] = {1, '1', 'p2', 'p3'}
    raised = False
    try:
        validate_role_manifest(roles_collision)
    except RoleManifestError:
        raised = True
    check('canonical patient-ID collision (int 1 vs str "1") rejected', raised)

    # whitespace-variant collision: ' p9' and 'p9' also collide once stripped.
    roles_collision2 = {k: set(v) for k, v in disjoint.items()}
    roles_collision2['locked_confirm'] = {'p9', ' p9', 'p10'}
    raised = False
    try:
        validate_role_manifest(roles_collision2)
    except RoleManifestError:
        raised = True
    check('canonical patient-ID collision (whitespace variant) rejected', raised)


def test_output_freshness():
    with tempfile.TemporaryDirectory() as tmp:
        fresh_path = os.path.join(tmp, 'new_dest')
        assert_fresh_output_dir(fresh_path)  # nonexistent -> fresh, must not raise
        check('nonexistent destination accepted as fresh', True)

        os.makedirs(os.path.join(tmp, 'empty_dest'))
        assert_fresh_output_dir(os.path.join(tmp, 'empty_dest'))
        check('empty existing destination accepted as fresh', True)

        stale_dir = os.path.join(tmp, 'stale_dest')
        os.makedirs(stale_dir)
        with open(os.path.join(stale_dir, 'train_log.jsonl'), 'w') as f:
            f.write('{}\n')
        raised = False
        try:
            assert_fresh_output_dir(stale_dir)
        except OutputNotFreshError:
            raised = True
        check('directory with stale result file rejected', raised)
        check('stale directory contents untouched (no auto-clean)',
              os.path.exists(os.path.join(stale_dir, 'train_log.jsonl')))

        # Fix 3 (G0.2A.2): genuinely fail-closed — ANY non-empty existing
        # directory is rejected in scientific mode, not just recognized names.
        def _make_and_expect_rejected(dirname, populate):
            d = os.path.join(tmp, dirname)
            os.makedirs(d)
            populate(d)
            raised_local = False
            try:
                assert_fresh_output_dir(d)
            except OutputNotFreshError:
                raised_local = True
            check('scientific mode rejects %s' % dirname, raised_local)

        _make_and_expect_rejected('unknown_text_file_dest',
                                   lambda d: open(os.path.join(d, 'notes.txt'), 'w').write('hello'))
        _make_and_expect_rejected('old_checkpointlike_dest',
                                   lambda d: open(os.path.join(d, 'model_v9_final_FINAL.pth'), 'wb').write(b'x'))
        _make_and_expect_rejected('nested_dir_dest',
                                   lambda d: os.makedirs(os.path.join(d, 'subdir')))
        _make_and_expect_rejected('recognized_result_dest',
                                   lambda d: open(os.path.join(d, 'checkpoint_manifest.json'), 'w').write('{}'))

        empty2 = os.path.join(tmp, 'empty_dest_2')
        os.makedirs(empty2)
        assert_fresh_output_dir(empty2)
        check('empty existing directory still accepted (Fix 3 unchanged for this case)', True)

        nonexist2 = os.path.join(tmp, 'never_created')
        assert_fresh_output_dir(nonexist2)
        check('nonexistent directory still accepted (Fix 3 unchanged for this case)', True)


def test_weight_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        weight_file = os.path.join(tmp, 'fake_weights.pth')
        with open(weight_file, 'wb') as f:
            f.write(b'not a real checkpoint, synthetic test bytes only')

        rec = record_weight_provenance(
            weight_file_path=weight_file, weight_enum='ResNet50_Weights.IMAGENET1K_V1',
            torchvision_version='0.15.0-synthetic-test', architecture_identifier='resnet50')
        check('weight provenance record has sha256', bool(rec['weight_file_sha256']))

        raised = False
        try:
            record_weight_provenance(weight_file_path=weight_file, weight_enum='', torchvision_version='x',
                                      architecture_identifier='resnet50', scientific_mode=True)
        except WeightProvenanceError:
            raised = True
        check('scientific mode rejects missing weight_enum', raised)

        raised = False
        try:
            record_weight_provenance(weight_file_path='/nonexistent/path.pth', weight_enum='e',
                                      torchvision_version='v', architecture_identifier='a', scientific_mode=True)
        except WeightProvenanceError:
            raised = True
        check('scientific mode rejects missing weight file', raised)


class ToyCritic(nn.Module):
    """Trainable stand-in for the verifier critic — takes the generator's
    fake output as input and must remain trainable through it."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4 * 16 * 16, 1)

    def forward(self, x):
        return self.fc(x.flatten(1))


def test_downstream_autograd_safe_forward():
    """Fix 1 (G0.2A.2): a fake generated under preserved_eval_forward must be
    usable as input to a critic that IS trained (backward + optimizer step)
    immediately afterward. This fails against the pre-fix inference_mode()
    implementation (PyTorch refuses to let an inference tensor participate
    in a graph that gets backpropagated) and passes against the no_grad()
    fix."""
    torch.manual_seed(1)
    generator = ToyBNNet()
    critic = ToyCritic()
    critic_opt = torch.optim.SGD(critic.parameters(), lr=0.1)
    x = torch.randn(8, 1, 16, 16)

    gen_state_before = canonical_tensor_state_hash(generator)

    with preserved_eval_forward(generator):
        fake = generator(x)

    raised = False
    try:
        critic_opt.zero_grad()
        out = critic(fake)
        loss = out.pow(2).mean()
        loss.backward()
    except RuntimeError:
        raised = True
    check('critic can backward through a preserved_eval_forward output (no RuntimeError)', not raised)

    grads = [p.grad for p in critic.parameters()]
    check('critic gradients are all present and finite',
          all(g is not None and torch.isfinite(g).all().item() for g in grads))
    check('critic gradients are nonzero (real training signal reached them)',
          any(g.abs().sum().item() > 0 for g in grads))
    check('generator parameters received NO gradient (fake was detached from autograd)',
          all(p.grad is None for p in generator.parameters()))

    critic_opt.step()
    check('generator state unchanged after downstream critic step',
          canonical_tensor_state_hash(generator) == gen_state_before)


class ToyParent(nn.Module):
    """Nested model with deliberately heterogeneous per-submodule .training
    flags, for Fix 2's submodule-mode-topology test."""

    def __init__(self):
        super().__init__()
        self.child_a = nn.BatchNorm1d(4)
        self.child_b = nn.BatchNorm1d(4)

    def forward(self, x):
        return self.child_b(self.child_a(x))


def _mode_vector_of(module):
    return [m.training for m in module.modules()]


def test_submodule_mode_topology_preserved():
    """Fix 2 (G0.2A.2): preserved_eval_forward must restore the EXACT
    per-submodule .training vector, not just the top-level flag."""
    parent = ToyParent()
    parent.train(True)
    parent.child_a.train(False)   # heterogeneous: parent=True, child_a=False, child_b=True
    parent.child_b.train(True)
    before = _mode_vector_of(parent)
    check('fixture actually has heterogeneous modes before the guard',
          len(set(before)) > 1)

    x = torch.randn(4, 4)
    with preserved_eval_forward(parent):
        parent(x)
    after_normal = _mode_vector_of(parent)
    check('exact per-submodule mode vector restored after normal exit', before == after_normal)

    # re-establish heterogeneous state, then force an exception inside the guard
    parent.train(True)
    parent.child_a.train(False)
    parent.child_b.train(True)
    before2 = _mode_vector_of(parent)
    raised = False
    try:
        with preserved_eval_forward(parent):
            parent(x)
            raise ValueError('deliberate')
    except ValueError:
        raised = True
    check('exception still propagates with heterogeneous modes', raised)
    after_exc = _mode_vector_of(parent)
    check('exact per-submodule mode vector restored after exception exit', before2 == after_exc)


def test_full_hash_schedule_source():
    """Fix 5 (G0.2A.2): AST/source-level check only — does not import or
    execute the real runner. Confirms the predeclared schedule constant is
    actually referenced by the class (not a dead constant), and that the
    trigger condition binds epoch + batch index + extra-step index together
    (so it cannot fire on every batch of a selected epoch)."""
    import ast
    path = os.path.join(ROOT, 'reproduction', 'method_dev', 'run_hardened_verifier_v2.py')
    src = open(path).read()
    tree = ast.parse(src, filename=path)

    check('PREDECLARED_FULL_HASH_EPOCHS is defined at module level',
          any(isinstance(n, ast.Assign) and any(
              isinstance(t, ast.Name) and t.id == 'PREDECLARED_FULL_HASH_EPOCHS' for t in n.targets)
              for n in tree.body))

    check('self.full_hash_epochs is bound to the predeclared constant (not a caller param)',
          'self.full_hash_epochs = PREDECLARED_FULL_HASH_EPOCHS' in src)
    check('constructor no longer accepts a caller-supplied snapshot_epochs parameter',
          'snapshot_epochs' not in src)

    check('trigger condition combines epoch schedule, batch index, and extra-step index',
          'epoch_is_scheduled and n_batches == 0 and extra_i == 0' in src)


def test_rng_seeding_source():
    """Fix 6 (G0.2A.2): AST/source-level check only — confirms every RNG
    domain the parity-sensitive original path seeds is also seeded here.
    Does NOT claim bitwise parity (that requires an actual GPU run)."""
    path = os.path.join(ROOT, 'reproduction', 'method_dev', 'run_hardened_verifier_v2.py')
    src = open(path).read()
    check('imports the `random` module', 'import random' in src)
    for token in ('torch.manual_seed(seed)', 'torch.cuda.manual_seed_all(seed)',
                  'torch.cuda.manual_seed(seed)', 'np.random.seed(seed)', 'random.seed(seed)'):
        check('source seeds: %s' % token, token in src)


def main():
    test_state_invariants()
    test_downstream_autograd_safe_forward()
    test_submodule_mode_topology_preserved()
    test_deterministic_loader()
    test_role_manifest()
    test_output_freshness()
    test_weight_provenance()
    test_full_hash_schedule_source()
    test_rng_seeding_source()
    n_pass = sum(1 for _, ok in RESULTS if ok)
    print('\n%d/%d checks passed' % (n_pass, len(RESULTS)))
    assert n_pass == len(RESULTS)
    print('ALL PROTOCOL_V2 TESTS PASS')


if __name__ == '__main__':
    main()
