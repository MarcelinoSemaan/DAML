#!/usr/bin/env python3
"""
EMBER Feature Extractor for DAML
Extracts 2568-dim EMBER features from PE files
"""
import sys
import json
import os
import numpy as np

def extract_features(file_path: str):
    """
    Extract EMBER features from a PE file.
    Returns 2568-dimensional feature vector.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Try ember first, then thrember
        features = None
        
        try:
            import ember
            # EMBER returns 2381 features, pad to 2568
            raw_features = ember.extract_features(file_path)
            if raw_features is None:
                raise ValueError("EMBER could not extract features - not a valid PE")
            features = np.pad(raw_features, (0, 2568 - len(raw_features)), mode='constant')
            
        except ImportError:
            try:
                import thrember
                # thrember should return 2568 features directly
                features = thrember.extract_features(file_path)
                if features is None:
                    raise ValueError("Thrember could not extract features")
                    
            except ImportError:
                raise ImportError("Neither ember nor thrember installed. Run: pip install git+https://github.com/elastic/ember.git")
        
        # Ensure correct shape and sanitize
        features = np.array(features, dtype=np.float32).flatten()
        if len(features) != 2568:
            raise ValueError(f"Expected 2568 features, got {len(features)}")
        
        # Replace NaN/Inf with 0 (EMBER sometimes has these)
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