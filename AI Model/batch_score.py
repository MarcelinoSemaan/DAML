#!/usr/bin/env python3
"""
Batch scoring for EmberLSTM.
Usage:
    python batch_score.py --input-folder "C:\SuspiciousFiles" --output results.jsonl
"""
import argparse
import json
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

# ── Config (must match training exactly) ──────────────────────────────────────
METADATA_KEYS = {'sha256', 'md5', 'appeared', 'label', 'avclass',
                 'feature_version', 'subset', 'source'}
MAX_LIST_LEN  = 256
TARGET_NF     = 64
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE   = DEVICE.type

# ── Import your extractor ────────────────────────────────────────────────────
from extract_corrected import extract_all_features, flatten_record

# ── Model (exact copy from training) ──────────────────────────────────────────
class EmberLSTM(nn.Module):
    def __init__(self, n_features: int, n_timesteps: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True,
                            dropout=0.3, bidirectional=True)
        self.attn = nn.Linear(512, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1).unsqueeze(-1)
        context = (lstm_out * weights).sum(dim=1)
        return self.classifier(context).squeeze(-1)

# ── Load artifacts ───────────────────────────────────────────────────────────
def load_artifacts(model_path, scaler_path, features_path):
    with open(features_path, 'rb') as f:
        feature_cols = pickle.load(f)
    col_index = {c: i for i, c in enumerate(feature_cols)}
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    n_cols = len(feature_cols)
    n_timesteps = max(1, math.ceil(n_cols / TARGET_NF))
    n_features = TARGET_NF
    pad_to = n_timesteps * n_features
    
    model = EmberLSTM(n_features, n_timesteps).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    return model, scaler, col_index, n_timesteps, n_features, pad_to

# ── Preprocess one record ────────────────────────────────────────────────────
def preprocess_record(record, col_index, scaler, n_timesteps, n_features, pad_to):
    flat = flatten_record(record)
    x = np.zeros(len(col_index), dtype=np.float32)
    for col, idx in col_index.items():
        val = flat.get(col, 0.0)
        if not np.isfinite(val):
            val = 0.0
        x[idx] = val
    
    x_scaled = scaler.transform(x.reshape(1, -1))[0]
    if len(x_scaled) < pad_to:
        x_scaled = np.pad(x_scaled, (0, pad_to - len(x_scaled)), constant_values=0.0)
    
    x_reshaped = x_scaled.reshape(n_timesteps, n_features)
    return torch.tensor(x_reshaped, dtype=torch.float32)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    default_dir = r"C:\Users\Dell\Downloads\DAML-LSTM-Scaler-version\DAML-LSTM-Scaler-version\AI Model"
    
    parser = argparse.ArgumentParser(description='Batch EmberLSTM Scoring')
    parser.add_argument('--input-folder', required=True, help='Folder containing PE files')
    parser.add_argument('--model',    default=str(Path(default_dir) / 'ember_lstm_best.pt'))
    parser.add_argument('--scaler',   default=str(Path(default_dir) / 'scaler.pkl'))
    parser.add_argument('--features', default=str(Path(default_dir) / 'feature_columns.pkl'))
    parser.add_argument('--output',   default='batch_results.jsonl')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    
    # Find all PE files
    input_folder = Path(args.input_folder)
    pe_files = list(input_folder.glob('*.exe')) + list(input_folder.glob('*.dll')) + list(input_folder.glob('*.sys'))
    print(f"Found {len(pe_files)} PE files in {input_folder}")
    
    if not pe_files:
        print("No PE files found. Exiting.")
        return
    
    # Load model
    model, scaler, col_index, n_ts, n_nf, pad_to = load_artifacts(
        args.model, args.scaler, args.features
    )
    
    # Process in batches
    results = []
    batch_records = []
    batch_files = []
    
    for pe_path in tqdm(pe_files, desc="Extracting features"):
        try:
            record = extract_all_features(pe_path)
            batch_records.append(record)
            batch_files.append(pe_path.name)
        except Exception as e:
            print(f"  ERROR extracting {pe_path.name}: {e}")
            continue
        
        # Score when batch is full
        if len(batch_records) >= args.batch_size:
            results.extend(score_batch(model, scaler, col_index, n_ts, n_nf, pad_to, 
                                       batch_records, batch_files, args.threshold))
            batch_records = []
            batch_files = []
    
    # Score remaining
    if batch_records:
        results.extend(score_batch(model, scaler, col_index, n_ts, n_nf, pad_to,
                                   batch_records, batch_files, args.threshold))
    
    # Save results
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    # Summary
    malicious = [r for r in results if r['prediction'] == 'malicious']
    benign = [r for r in results if r['prediction'] == 'benign']
    errors = [r for r in results if r['prediction'] == 'error']
    
    print(f"\n{'='*50}")
    print(f"BATCH SCORING COMPLETE")
    print(f"{'='*50}")
    print(f"Total files:    {len(results)}")
    print(f"Malicious:      {len(malicious)} ({len(malicious)/len(results)*100:.1f}%)")
    print(f"Benign:         {len(benign)} ({len(benign)/len(results)*100:.1f}%)")
    print(f"Errors:         {len(errors)}")
    print(f"Results saved:  {args.output}")
    
    # Top 10 most suspicious
    if malicious:
        print(f"\nTop 10 most suspicious files:")
        sorted_mal = sorted(malicious, key=lambda x: x['malware_probability'], reverse=True)
        for r in sorted_mal[:10]:
            print(f"  {r['malware_probability']:.4f} | {r['file']}")

def score_batch(model, scaler, col_index, n_ts, n_nf, pad_to, records, filenames, threshold):
    batch_results = []
    
    # Preprocess all records
    tensors = []
    for record in records:
        try:
            t = preprocess_record(record, col_index, scaler, n_ts, n_nf, pad_to)
            tensors.append(t)
        except Exception as e:
            batch_results.append({
                'file': filenames[len(batch_results)],
                'error': str(e),
                'prediction': 'error'
            })
            tensors.append(None)
    
    # Filter out errors
    valid_indices = [i for i, t in enumerate(tensors) if t is not None]
    valid_tensors = [tensors[i] for i in valid_indices]
    
    if not valid_tensors:
        return batch_results
    
    # Batch inference
    xb = torch.stack(valid_tensors).to(DEVICE)
    with torch.no_grad():
        with torch.amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == 'cuda')):
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
    
    # Build results
    prob_idx = 0
    for i in range(len(records)):
        if i in valid_indices:
            prob = float(probs[prob_idx])
            pred = 'malicious' if prob > threshold else 'benign'
            batch_results.append({
                'file': filenames[i],
                'sha256': records[i].get('sha256', 'unknown'),
                'malware_probability': round(prob, 4),
                'prediction': pred
            })
            prob_idx += 1
        # else: error result already added
    
    return batch_results

if __name__ == '__main__':
    main()