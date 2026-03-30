#!/usr/bin/env python3
"""
DAML Feature Extractor
Extracts 2568 features from PE files using thrember for the EmberLSTM model.
"""

import sys
import json
import numpy as np

# Add thrember to path if needed (adjust based on your installation)
try:
    import thrember
except ImportError:
    # Try common alternative install locations
    import site
    import os
    # Check if thrember is in the same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(script_dir, 'thrember')):
        sys.path.insert(0, script_dir)
        import thrember


def extract_features(file_path: str):
    """
    Extract 2568 features from a PE file using thrember.
    
    Args:
        file_path: Path to the PE file (.exe, .dll, etc.)
        
    Returns:
        list: 2568 float features as JSON array
    """
    try:
        # Initialize feature extractor
        extractor = thrember.PEFeatureExtractor()
        
        # Extract raw features from file
        raw_features = extractor.extract(file_path)
        
        # Convert to numpy array
        features = np.array(raw_features, dtype=np.float32)
        
        # Ensure exact size: 2568 features (24 timesteps × 107 features)
        target_size = 2568
        
        if len(features) < target_size:
            # Pad with zeros if too short
            features = np.pad(features, (0, target_size - len(features)), mode='constant')
        elif len(features) > target_size:
            # Truncate if too long
            features = features[:target_size]
        
        # Clean invalid values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Validate
        if not np.all(np.isfinite(features)):
            raise ValueError("Features contain invalid values after cleaning")
        
        # Output as JSON
        print(json.dumps(features.tolist()))
        return 0
        
    except FileNotFoundError:
        print(json.dumps({"error": f"File not found: {file_path}"}), file=sys.stderr)
        return 1
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python extract_features.py <path_to_pe_file>"}), file=sys.stderr)
        sys.exit(1)
    
    sys.exit(extract_features(sys.argv[1]))