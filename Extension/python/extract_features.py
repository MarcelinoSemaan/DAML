#!/usr/bin/env python3
"""
EMBER Feature Extractor for DAML
Extracts 2568-dim EMBER features from PE files using thrember (EMBER2024)
Compatible with Python 3.11, no LIEF dependency
"""
import sys
import json
import os
import numpy as np
import hashlib

# Patch hashlib to auto-encode strings (fixes compatibility issues)
_original_md5 = hashlib.md5
def _patched_md5(data=b'', *args, **kwargs):
    if isinstance(data, str):
        data = data.encode('utf-8', errors='ignore')
    return _original_md5(data, *args, **kwargs)
hashlib.md5 = _patched_md5

_original_sha256 = hashlib.sha256
def _patched_sha256(data=b'', *args, **kwargs):
    if isinstance(data, str):
        data = data.encode('utf-8', errors='ignore')
    return _original_sha256(data, *args, **kwargs)
hashlib.sha256 = _patched_sha256

def extract_features(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        import thrember
        
        # Read file as bytes
        with open(file_path, 'rb') as f:
            bytez = f.read()
        
        extractor = thrember.PEFeatureExtractor()
        raw_feats = extractor.raw_features(bytez)
        
        if raw_feats is None:
            raise ValueError("Could not extract features")
        
        processed = extractor.process_raw_features(raw_feats)
        features = np.array(processed, dtype=np.float32)
        
        # Pad/truncate to exactly 2568 dimensions
        if len(features) < 2568:
            features = np.pad(features, (0, 2568 - len(features)), mode='constant')
        else:
            features = features[:2568]
        
        # Clean up NaN/Inf values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(json.dumps(features.tolist()))
        return 0
        
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {str(e)}"}), file=sys.stderr)
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: extract_features.py <pe_file>"}), file=sys.stderr)
        sys.exit(1)
    
    sys.exit(extract_features(sys.argv[1]))