"""Calibrate the cheap proxy re-identification score against the expensive 10-seed protocol.

The full protocol re-trains a Siamese network ten times (~10 h per generator). The proxy (proxy_reid.py)
measures verification AUC in the feature space of a frozen ImageNet ResNet-50 in minutes. Before trusting it
for fast iteration we must show it correlates with the real 10-seed AUC on the points we have already
measured -- the discipline-a requirement from the plan.

Usage (single point):
    python calibrate_proxy.py --tag run_1 \
        --checkpoint ./archive/train_prichexy_net_run_1/generator_lowest_total_loss.pth \
        --real 0.6038

Usage (batch, reads existing proxy logs):
    python calibrate_proxy.py --from-logs
"""

import argparse
import glob
import json
import os
import re

import numpy as np


# Real 10-seed Re-ID AUC (mean) we already measured with the full protocol.
REAL_10SEED = {
    'run_1': 0.6038,
    'run_2': 0.6223,
    'run_3': 0.7063,
    'run_4': 0.6060,
    'baseline_none': 0.8015,
    'randomwarp': 0.8174,
}


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float('nan')
    if np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float('nan')
    return float(spearmanr(x, y).statistic)


def parse_proxy_log(path):
    """Pull PROXY_AUC out of a proxy_reid.py log file."""
    with open(path) as f:
        for line in f:
            m = re.search(r'PROXY_AUC:\s*([0-9.]+)', line)
            if m:
                return float(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('--from-logs', action='store_true',
                    help='Read PROXY_AUC from per-point log files instead of running the proxy.')
    ap.add_argument('--logdir', default='/workspace',
                    help='Directory containing proxy_<tag>.log files for each REAL_10SEED tag.')
    ap.add_argument('--checkpoint', default=None, help='Single-checkpoint mode.')
    args = ap.parse_args()

    tags, proxies, reals = [], [], []

    if args.checkpoint is not None:
        print('Single-calibration mode: run this with --from-logs and per-tag logs instead.')
        return

    if args.from_logs:
        for tag, real in REAL_10SEED.items():
            log = os.path.join(args.logdir, f'proxy_{tag}.log')
            if os.path.exists(log):
                val = parse_proxy_log(log)
                if val is not None:
                    tags.append(tag)
                    proxies.append(val)
                    reals.append(real)
                    print(f'  {tag:16s} proxy={val:.4f}  real10={real:.4f}')
                else:
                    print(f'  {tag:16s} log exists but PROXY_AUC not found')
            else:
                print(f'  {tag:16s} no log at {log}')
    else:
        ap.error('Provide --from-logs (or a --checkpoint with a tag).')

    print()
    print(f'Calibration points: {len(tags)}')
    if len(tags) < 3:
        print('Need at least 3 points to estimate correlation.')
        return

    x, y = np.asarray(proxies, float), np.asarray(reals, float)
    print(f'Pearson  r = {pearson(x, y):+.3f}')
    print(f'Spearman ρ = {spearman(x, y):+.3f}')
    print('\nProxy → real 10-seed linear fit (least squares):')
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    print(f'  real10 = {slope:+.3f} * proxy {intercept:+.3f}')
    print('\nVerdict: correlations above ~0.85 justify using the proxy for fast selection;')
    print('below ~0.7 the proxy is too noisy and the full 10-seed protocol stays mandatory.')


if __name__ == '__main__':
    main()