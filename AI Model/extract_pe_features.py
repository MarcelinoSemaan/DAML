#!/usr/bin/env python3
"""
Extract EMBER features using the official elastic/ember library.
Usage:
    python extract_ember.py --input "C:\Windows\System32\notepad.exe" --output notepad_ember.jsonl
"""
import argparse
import json
import hashlib
from pathlib import Path

try:
    import ember
except ImportError:
    raise ImportError("ember not installed. Run: pip install git+https://github.com/elastic/ember.git")

def extract_ember_features(pe_path: Path):
    # Use the official EMBER raw feature extractor (returns nested dict)
    extractor = ember.PEFeatureExtractor()
    raw = extractor.raw_features(str(pe_path))
    
    if raw is None:
        raise RuntimeError("EMBER failed to extract features (possibly not a valid PE)")
    
    # Add metadata that inference.py expects
    data = open(pe_path, 'rb').read()
    sha256 = hashlib.sha256(data).hexdigest()
    md5 = hashlib.md5(data).hexdigest()
    
    record = {
        'sha256': sha256,
        'md5': md5,
        'appeared': '',
        'label': -1,           # unknown / unlabeled for inference
        'avclass': 'unknown',
        'feature_version': raw.get('feature_version', 2),
        'subset': 'test',
        'source': 'ember_official',
    }
    
    # Merge the EMBER feature dict (histogram, byteentropy, strings, general, header, section, imports, exports, datadirectories)
    record.update(raw)
    
    return record

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Path to PE file')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    args = parser.parse_args()
    
    pe_path = Path(args.input)
    if not pe_path.exists():
        raise FileNotFoundError(pe_path)
    
    print(f"Extracting official EMBER features from {pe_path} ...")
    record = extract_ember_features(pe_path)
    
    with open(args.output, 'w') as f:
        f.write(json.dumps(record) + '\n')
    
    print(f"Saved to {args.output}")
    
    # Quick sanity check: flatten and count keys to see if it matches 689
    # (We import the same flatten logic from inference to verify)
    import numpy as np
    METADATA_KEYS = {'sha256', 'md5', 'appeared', 'label', 'avclass',
                     'feature_version', 'subset', 'source'}
    MAX_LIST_LEN = 256
    
    def flatten(record, prefix=''):
        out = {}
        for k, v in record.items():
            fk = f"{prefix}.{k}" if prefix else k
            if fk in METADATA_KEYS:
                continue
            if isinstance(v, dict):
                out.update(flatten(v, fk))
            elif isinstance(v, list):
                for i, item in enumerate(v[:MAX_LIST_LEN]):
                    if isinstance(item, bool):
                        out[f"{fk}.{i}"] = float(item)
                    elif isinstance(item, (int, float)):
                        val = float(item)
                        out[f"{fk}.{i}"] = 0.0 if not np.isfinite(val) else val
            elif isinstance(v, bool):
                out[fk] = float(v)
            elif isinstance(v, (int, float)):
                val = float(v)
                out[fk] = 0.0 if not np.isfinite(val) else val
        return out
    
    flat_keys = sorted(flatten(record).keys())
    print(f"Flattened feature count: {len(flat_keys)} (model expects 689)")
    
    # Load expected columns and compare
    import pickle
    feat_path = Path(r"C:\Users\Dell\Downloads\DAML-LSTM-Scaler-version\DAML-LSTM-Scaler-version\AI Model\feature_columns.pkl")
    if feat_path.exists():
        with open(feat_path, 'rb') as f:
            expected = set(pickle.load(f))
        actual = set(flat_keys)
        missing = expected - actual
        extra = actual - expected
        if missing:
            print(f"WARNING: {len(missing)} expected features MISSING from EMBER output")
            print(f"  Examples: {list(missing)[:5]}")
        if extra:
            print(f"WARNING: {len(extra)} unexpected EXTRA features in EMBER output")
            print(f"  Examples: {list(extra)[:5]}")
        if not missing and not extra:
            print("PERFECT MATCH: All 689 features align with training schema!")

if __name__ == '__main__':
    main()




    