"""Run N retrain_SNN runs with different seeds and compute mean ± std AUC (paper protocol).

Usage:
    python run_snn_multiseed.py --n_runs 10 --checkpoint ./archive/.../generator_lowest_total_loss.pth \
        --out_dir ./archive/retrain_snn_runs_total
"""
import os
import json
import shutil
import subprocess
import argparse


def make_config(base_config, experiment_description, checkpoint, seed):
    cfg = dict(base_config)
    cfg['experiment_description'] = experiment_description
    if checkpoint is not None:
        cfg['perturbation_model_file'] = checkpoint
    cfg['seed'] = seed
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--checkpoint', default=None,
                        help='Generator checkpoint. Omit for perturbation_type "none" baselines.')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--base_config', default='./config_files/config_retrainSNN.json')
    parser.add_argument('--start_seed', type=int, default=0)
    args = parser.parse_args()

    with open(args.base_config, 'r') as f:
        base_config = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    run_configs = []
    for i in range(args.n_runs):
        seed = args.start_seed + i
        exp_name = 'retrain_snn_seed{}'.format(seed)
        cfg = make_config(base_config, exp_name, args.checkpoint, seed)
        cfg_path = os.path.join('config_files', exp_name + '.json')
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        run_configs.append((exp_name, cfg_path, seed))

    for exp_name, cfg_path, seed in run_configs:
        print('\n========== RUN seed={} exp={} =========='.format(seed, exp_name))
        subprocess.run(['python', 'retrain_SNN.py', '--config_path', './config_files/', '--config', exp_name + '.json'],
                       check=True)
        # Collect the results file into out_dir
        src = os.path.join('./archive/', exp_name, exp_name + '_results.txt')
        if os.path.exists(src):
            dst = os.path.join(args.out_dir, exp_name + '_results.txt')
            shutil.copy(src, dst)
            print('Collected:', dst)
        # Clean up the generated config
        os.remove(cfg_path)

    # Compute mean ± std directly from the collected AUC lines (robust to file format)
    aucs = []
    for fname in sorted(os.listdir(args.out_dir)):
        if not fname.endswith('_results.txt'):
            continue
        with open(os.path.join(args.out_dir, fname), 'r') as f:
            for line in f:
                if line.startswith('AUC:'):
                    aucs.append(float(line.split(':')[1].strip()))
                    break
    if not aucs:
        raise RuntimeError('No AUC values found in ' + args.out_dir)
    import numpy as np
    aucs = np.array(aucs)
    mean_result = aucs.mean()
    std_result = aucs.std()
    print('\n========== SUMMARY ({}) =========='.format(args.out_dir))
    print('N runs: {}'.format(len(aucs)))
    print('Per-run AUC: {}'.format(np.round(aucs, 4)))
    print('AUC mean ± std: {:.4f} ± {:.4f}'.format(mean_result, std_result))
    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        f.write('N_runs: {}\n'.format(len(aucs)))
        f.write('AUC_mean: {}\n'.format(mean_result))
        f.write('AUC_std: {}\n'.format(std_result))
        f.write('Per_run: {}\n'.format(list(aucs)))


if __name__ == '__main__':
    main()
