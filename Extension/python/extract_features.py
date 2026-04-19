#!/usr/bin/env python3
"""
EMBER Feature Extractor for DAML (64-feature version)
Extracts 1536-dim features (64 × 24) to match checkpoint
"""
import sys
import json
import os
import numpy as np
import hashlib

# Patch hashlib for thrember compatibility
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
        
        with open(file_path, 'rb') as f:
            bytez = f.read()
        
        extractor = thrember.PEFeatureExtractor()
        raw_feats = extractor.raw_features(bytez)
        
        if raw_feats is None:
            raise ValueError("Could not extract features")
        
        # Get full 107-dim features
        processed = extractor.process_raw_features(raw_feats)
        features = np.array(processed, dtype=np.float32)
        
        # Ensure we have 2568 features (107 × 24)
        if len(features) < 2568:
            features = np.pad(features, (0, 2568 - len(features)), mode='constant')
        else:
            features = features[:2568]
        
        # Reshape to (24, 107) — 24 timesteps, 107 features each
        features = features.reshape(24, 107)
        
        # SELECT first 64 features per timestep (not truncate total!)
        # This preserves the structure: 24 timesteps × 64 features = 1536
        features = features[:, :64]  # Take first 64 of each 107-dim timestep
        
        # Flatten back to 1536
        features = features.flatten()
        
        # Clean
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