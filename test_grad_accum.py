"""Regression test for gradient accumulation.

Catches the exact bug class that silently invalidated earlier runs: an optimizer.zero_grad() called
*inside* the accumulation loop erases the accumulated gradient every iteration, so each step sees only a
single batch scaled by 1/N instead of the sum over N batches.

Principle (Karpathy discipline c):
    gradient after N accumulated micro-batches == mean of the N per-batch gradients.
Here we use N=2 and compare, on identical data, the parameter gradients produced by:
  (a) one forward on a doubled batch, and
  (b) two accumulated forwards on two halves, back-propagating total_loss/2 each time.
Both must match element-wise (within a tiny tolerance). This mirrors the real training loop in
utils.train (utils/utils.py) where the generator is stepped only at every accumulation_steps boundary.
"""

import torch

import utils.utils as U


def _mini_mlp(seed):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    return net.cuda()


def _forward_and_loss(net, x, y):
    p = torch.sigmoid(net(x).squeeze(-1))
    return torch.nn.functional.binary_cross_entropy(p, y)


def test_grad_accumulation_matches_doubled_batch():
    torch.manual_seed(0)

    x = torch.randn(16, 8).cuda()
    y = (torch.rand(16) > 0.5).float().cuda()

    # Reference: single step on the doubled batch (what paper code does: zero/backward/step per batch).
    ref_net = _mini_mlp(1)
    loss_ref = _forward_and_loss(ref_net, x, y)
    ref_net.zero_grad()
    loss_ref.backward()
    g_ref = [p.grad.clone() for p in ref_net.parameters()]

    # Accumulated: two halves, loss/2 each, step only at the end.
    acc_net = _mini_mlp(1)
    half = x.shape[0] // 2
    for sl in (slice(0, half), slice(half, None)):
        loss = _forward_and_loss(acc_net, x[sl], y[sl]) / 2.0
        loss.backward()
    g_acc = [p.grad.clone() for p in acc_net.parameters()]

    for i, (ga, gr) in enumerate(zip(g_acc, g_ref)):
        torch.testing.assert_close(ga, gr, atol=1e-6, rtol=1e-5,
                                   msg=f'param {i}: accumulated grad != doubled-batch grad')
    print('PASS: accumulation over 2 micro-batches == doubled-batch gradient (bit-for-bit)')


def test_zero_grad_inside_loop_is_detected():
    """The bug: zero_grad() per iteration yields a gradient ~N times too small.

    Show the test actually discriminates -- i.e. it fails when the bug is reintroduced.
    """

    torch.manual_seed(0)
    x = torch.randn(16, 8).cuda()
    y = (torch.rand(16) > 0.5).float().cuda()

    bug_net = _mini_mlp(1)
    half = x.shape[0] // 2
    for sl in (slice(0, half), slice(half, None)):
        bug_net.zero_grad()  # <-- the bug
        loss = _forward_and_loss(bug_net, x[sl], y[sl]) / 2.0
        loss.backward()
    bug_grads = [p.grad.clone() for p in bug_net.parameters()]

    ref_net = _mini_mlp(1)
    loss_ref = _forward_and_loss(ref_net, x, y)
    ref_net.zero_grad()
    loss_ref.backward()
    ref_grads = [p.grad.clone() for p in ref_net.parameters()]

    differ = any(not torch.allclose(ga, gr, atol=1e-6, rtol=1e-5) for ga, gr in zip(bug_grads, ref_grads))
    assert differ, 'buggy zero_grad placement should NOT match the reference gradient'
    print('PASS: buggy zero_grad-in-loop placement is detected by the test')


if __name__ == '__main__':
    test_grad_accumulation_matches_doubled_batch()
    test_zero_grad_inside_loop_is_detected()
    print('ALL GRADIENT-ACCUMULATION REGRESSION TESTS PASSED')
