#!/usr/bin/env python3
"""
Find optimal classification threshold using labeled test data.
Usage:
    python find_threshold.py --input your_test_predictions.jsonl
"""
import argparse
import json
import numpy as np
from sklearn.metrics import (precision_recall_curve, f1_score, accuracy_score,
                             roc_auc_score, confusion_matrix)

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
    
    if not labels:
        print("No labeled samples found. Cannot optimize threshold.")
        return
    
    labels = np.array(labels)
    probs = np.array(probs)
    
    print(f"Evaluating {len(labels)} labeled samples...")
    print(f"Baseline (threshold=0.5): Acc={accuracy_score(labels, probs>0.5):.4f}, "
          f"F1={f1_score(labels, probs>0.5):.4f}, AUC={roc_auc_score(labels, probs):.4f}")
    
    # F1 sweep
    print(f"\n{'='*60}")
    print("THRESHOLD SWEEP (F1 optimization)")
    print(f"{'='*60}")
    
    best_f1, best_t = 0, 0.5
    best_acc, best_prec, best_rec = 0, 0, 0
    
    print(f"{'Threshold':>10} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
    print("-" * 60)
    
    for t in np.arange(0.05, 0.95, 0.05):
        preds = (probs > t).astype(int)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        
        # Manual precision/recall
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"{t:>10.2f} | {acc:>8.4f} | {prec:>9.4f} | {rec:>6.4f} | {f1:>6.4f}")
        
        if f1 > best_f1:
            best_f1, best_t = f1, t
            best_acc, best_prec, best_rec = acc, prec, rec
    
    print(f"\n{'='*60}")
    print(f"OPTIMAL THRESHOLD: {best_t:.2f}")
    print(f"  F1 Score:      {best_f1:.4f}")
    print(f"  Accuracy:      {best_acc:.4f}")
    print(f"  Precision:     {best_prec:.4f}")
    print(f"  Recall:        {best_rec:.4f}")
    print(f"{'='*60}")
    
    # Precision-Recall at specific operating points
    print(f"\nOPERATING POINTS:")
    prec_curve, rec_curve, thresh_curve = precision_recall_curve(labels, probs)
    
    # 99% precision point
    idx_99 = np.where(prec_curve >= 0.99)[0]
    if len(idx_99) > 0:
        i = idx_99[0]
        t = thresh_curve[i] if i < len(thresh_curve) else 1.0
        print(f"  99% Precision: threshold={t:.4f}, recall={rec_curve[i]:.4f}")
    
    # 95% recall point
    idx_95 = np.where(rec_curve >= 0.95)[0]
    if len(idx_95) > 0:
        i = idx_95[-1]
        t = thresh_curve[i] if i < len(thresh_curve) else 1.0
        print(f"  95% Recall:    threshold={t:.4f}, precision={prec_curve[i]:.4f}")
    
    # Confusion matrix at best threshold
    best_preds = (probs > best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, best_preds).ravel()
    print(f"\nConfusion Matrix (threshold={best_t:.2f}):")
    print(f"                 Predicted")
    print(f"                 Benign  Malicious")
    print(f"Actual Benign    {tn:>6}  {fp:>9}")
    print(f"       Malicious {fn:>6}  {tp:>9}")

if __name__ == '__main__':
    main()



    