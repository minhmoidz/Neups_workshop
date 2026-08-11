import json
import hashlib
import subprocess
from datetime import datetime, timezone

def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def git_head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

def main():
    clf = json.load(open('research_agent/03H_corrected_classification.json'))
    seg = json.load(open('research_agent/03H_corrected_segmentation.json'))
    reid = json.load(open('research_agent/03D_corrected_baseline_reid_summary_auditable.json'))

    out = {
        'generator_path': clf['generator_path'],
        'generator_sha256': clf['generator_sha256'],
        'transform_mode': 'corrected',
        'mu': 0.01,
        'stochastic_lambda': 0.0,
        'privacy': {
            'reid_mean': reid['mean_test_auc_source'],
            'reid_sample_sd': reid['std_test_auc_source'],
            'reid_median': reid['median_test_auc_source'],
            'reid_max': reid['max_test_auc_source'],
            'n_valid_attackers': reid['n_valid'],
            'source': 'research_agent/03D_corrected_baseline_reid_summary_auditable.json',
        },
        'classification': {
            'auc_by_label': clf['auc_by_label'],
            'mean_auc_14': clf['mean_auc_14'],
            'bootstrap_95ci_mean_auc_14': clf['bootstrap_95ci_mean_auc_14'],
            'classifier_checkpoint': clf['classifier_checkpoint'],
            'classifier_sha256': clf['classifier_sha256'],
            'classifier_architecture': clf['classifier_architecture'],
            'dataset_split': clf['dataset_split'],
            'dataset_split_sha256': clf['dataset_split_sha256'],
            'eval_config_path': clf['eval_config_path'],
            'eval_config_sha256': clf['eval_config_sha256'],
        },
        'segmentation': {
            'dice': seg['aggregates']['dice'],
            'iou': seg['aggregates']['iou'],
            'hd95': seg['aggregates']['hd95'],
            'bootstrap_95ci': seg['bootstrap_95ci'],
            'segmentation_checkpoint': seg['segmentation_checkpoint'],
            'segmentation_sha256': seg['segmentation_sha256'],
            'segmentation_architecture': seg['segmentation_architecture'],
            'anatomical_targets': seg['anatomical_targets'],
            'mask_metadata_sha256': seg['mask_metadata_sha256'],
            'dataset_split': seg['dataset_split'],
            'dataset_split_sha256': seg['dataset_split_sha256'],
        },
        'evaluation_config_hashes': {
            'config_anonymization_baseline_corrected.json': sha256('config_files/config_anonymization_baseline_corrected.json'),
            'config_eval_classifier_corrected_baseline.json': sha256('config_files/config_eval_classifier_corrected_baseline.json'),
            'eval_classifier.py': sha256('eval_classifier.py'),
            'chexnet/eval_model.py': sha256('chexnet/eval_model.py'),
            'eval_seg.py': sha256('eval_seg.py'),
            'research_agent/eval_seg_percase.py': sha256('research_agent/eval_seg_percase.py'),
            'networks/UNet_PriCheXyNet.py': sha256('networks/UNet_PriCheXyNet.py'),
            'networks/UNetSeg.py': sha256('networks/UNetSeg.py'),
        },
        'git_commit': git_head(),
        'created': datetime.now(timezone.utc).isoformat(),
    }
    with open('research_agent/03H_corrected_baseline_utility_summary.json', 'w') as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print('privacy: %.6f +/- %.6f (median %.6f, max %.6f)' % (
        out['privacy']['reid_mean'], out['privacy']['reid_sample_sd'],
        out['privacy']['reid_median'], out['privacy']['reid_max']))
    print('classification mean_auc_14 = %.6f' % out['classification']['mean_auc_14'])
    print('segmentation Dice=%.5f IoU=%.5f HD95=%.5f' % (
        out['segmentation']['dice'], out['segmentation']['iou'], out['segmentation']['hd95']))
    print('wrote research_agent/03H_corrected_baseline_utility_summary.json')

if __name__ == '__main__':
    main()
