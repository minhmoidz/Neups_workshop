"""STEP 8R — Pair and Split Forensics

Compares upstream PriCheXy-Net image_pairs/ files with current Neups_workshop image_pairs/.
Computes hashes, byte sizes, line counts, balance, patient and image overlap.
Does NOT open TEST images.
"""

import os
import hashlib
import json
import pandas as pd


UPSTREAM_DIR = '/home/minhtt/PriCheXy-Net_upstream_reproduction/image_pairs'
CURRENT_DIR = '/home/minhtt/Neups_workshop/image_pairs'

FILES = [
    'image_pairs_training_10000.txt',
    'image_pairs_validation_2000.txt',
    'image_pairs_testing_5000.txt',
    'train_val_list.txt',
    'test_list.txt'
]


def analyze_pairs(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    pos_count = 0
    neg_count = 0
    images = set()
    patients = set()
    
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            img1, img2, label = parts[0], parts[1], int(float(parts[2]))
            if label == 1:
                pos_count += 1
            elif label == 0:
                neg_count += 1
            images.add(img1)
            images.add(img2)
            # Patient ID is the first 8 digits in NIH format: 00000001_000.png
            patients.add(img1.split('_')[0])
            patients.add(img2.split('_')[0])
            
    return {
        'total_pairs': len(lines),
        'positive_pairs': pos_count,
        'negative_pairs': neg_count,
        'positive_ratio': float(pos_count / len(lines)) if lines else 0.0,
        'unique_images': len(images),
        'unique_patients': len(patients),
        'image_set': images,
        'patient_set': patients,
    }


def main():
    report = {}
    pair_summaries = {}
    
    for fn in FILES:
        p_up = os.path.join(UPSTREAM_DIR, fn)
        p_cur = os.path.join(CURRENT_DIR, fn)
        
        with open(p_up, 'rb') as f:
            bytes_up = f.read()
        with open(p_cur, 'rb') as f:
            bytes_cur = f.read()
            
        sha_up = hashlib.sha256(bytes_up).hexdigest()
        sha_cur = hashlib.sha256(bytes_cur).hexdigest()
        
        lines_up = [l for l in bytes_up.decode('utf-8').strip().split('\n') if l.strip()]
        lines_cur = [l for l in bytes_cur.decode('utf-8').strip().split('\n') if l.strip()]
        
        report[fn] = {
            'upstream_path': p_up,
            'current_path': p_cur,
            'byte_size_upstream': len(bytes_up),
            'byte_size_current': len(bytes_cur),
            'line_count_upstream': len(lines_up),
            'line_count_current': len(lines_cur),
            'sha256_upstream': sha_up,
            'sha256_current': sha_cur,
            'exact_byte_equality': (bytes_up == bytes_cur),
        }
        
        if 'image_pairs' in fn:
            stat_up = analyze_pairs(p_up)
            stat_cur = analyze_pairs(p_cur)
            pair_summaries[fn] = {
                'upstream': {
                    'total_pairs': stat_up['total_pairs'],
                    'positive_pairs': stat_up['positive_pairs'],
                    'negative_pairs': stat_up['negative_pairs'],
                    'positive_ratio': stat_up['positive_ratio'],
                    'unique_images': stat_up['unique_images'],
                    'unique_patients': stat_up['unique_patients'],
                },
                'current': {
                    'total_pairs': stat_cur['total_pairs'],
                    'positive_pairs': stat_cur['positive_pairs'],
                    'negative_pairs': stat_cur['negative_pairs'],
                    'positive_ratio': stat_cur['positive_ratio'],
                    'unique_images': stat_cur['unique_images'],
                    'unique_patients': stat_cur['unique_patients'],
                }
            }
            report[fn]['stats'] = pair_summaries[fn]

    # Split overlap analysis
    train_stat = analyze_pairs(os.path.join(UPSTREAM_DIR, 'image_pairs_training_10000.txt'))
    val_stat = analyze_pairs(os.path.join(UPSTREAM_DIR, 'image_pairs_validation_2000.txt'))
    test_stat = analyze_pairs(os.path.join(UPSTREAM_DIR, 'image_pairs_testing_5000.txt'))
    
    overlap_analysis = {
        'train_val_image_overlap': len(train_stat['image_set'].intersection(val_stat['image_set'])),
        'train_val_patient_overlap': len(train_stat['patient_set'].intersection(val_stat['patient_set'])),
        'train_test_image_overlap': len(train_stat['image_set'].intersection(test_stat['image_set'])),
        'train_test_patient_overlap': len(train_stat['patient_set'].intersection(test_stat['patient_set'])),
        'val_test_image_overlap': len(val_stat['image_set'].intersection(test_stat['image_set'])),
        'val_test_patient_overlap': len(val_stat['patient_set'].intersection(test_stat['patient_set'])),
    }
    
    output = {
        'files_comparison': report,
        'pair_statistics': pair_summaries,
        'split_overlap_analysis': overlap_analysis,
    }
    
    out_path = '/home/minhtt/Neups_workshop/research_agent/pair_file_hash_comparison.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
        
    print(f"Saved pair comparison to {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
