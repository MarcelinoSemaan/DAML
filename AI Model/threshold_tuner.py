#!/usr/bin/env python3
"""
Finds the optimal classification threshold.
Usage:
    python3 threshold_tuner.py --input predictions.jsonl
"""
import argparse
import json
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='JSONL with labels and malware_probability')
    args = parser.parse_args()
    
    labels, probs = [], []
    with open(args.input, 'r') as f:
        for line in f:
            r = json.loads(line)
            if r.get('label', -1) != -1:
                labels.append(1 if r['label'] > 0 else 0)
                probs.append(r['malware_probability'])
    
    labels = np.array(labels)
    probs = np.array(probs)
    
    # F1 sweep
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        preds = (probs > t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    
    print(f"Best F1 threshold: {best_t:.2f} (F1={best_f1:.4f})")
    
    # Precision-Recall curve data
    precision, recall, pr_thresh = precision_recall_curve(labels, probs)
    
    # Find threshold for 95% recall
    idx_95 = np.where(recall >= 0.95)[0]
    if len(idx_95) > 0:
        t_95 = pr_thresh[idx_95[-1]]
        p_95 = precision[idx_95[-1]]
        print(f"For 95% recall: threshold={t_95:.3f}, precision={p_95:.3f}")
    
    # Find threshold for 99% precision
    idx_99 = np.where(precision >= 0.99)[0]
    if len(idx_99) > 0:
        t_99 = pr_thresh[idx_99[0]]
        r_99 = recall[idx_99[0]]
        print(f"For 99% precision: threshold={t_99:.3f}, recall={r_99:.3f}")

if __name__ == '__main__':
    main()