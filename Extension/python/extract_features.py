#!/usr/bin/env python3
r"""
DAML Feature Extractor - FIXED to match training schema.
Usage: python extract_features_fixed.py <pe_file_or_json>
Outputs: JSON array of floats aligned to training schema
"""
import sys
import json
import math
from pathlib import Path
import numpy as np
import pickle

# ------------------------------------------------------------------------------
FEATURE_COLS_PATH = Path(
    r"C:\Users\Dell\Downloads\DAML-LSTM-Scaler-version\DAML-LSTM-Scaler-version\AI Model\feature_columns.pkl"
)

METADATA_KEYS = {"sha256", "md5", "appeared", "label", "avclass",
                 "feature_version", "subset", "source"}
MAX_LIST_LEN = 256

# ------------------------------------------------------------------------------
# THIS MUST MATCH DAML.PY EXACTLY - no imports special-casing
def flatten_record(record, prefix=""):
    out = {}
    for k, v in record.items():
        fk = f"{prefix}.{k}" if prefix else k
        if fk in METADATA_KEYS:
            continue
        if isinstance(v, dict):
            # Simple recursion - NO imports special-casing
            out.update(flatten_record(v, fk))
        elif isinstance(v, list):
            for i, item in enumerate(v[:MAX_LIST_LEN]):
                if isinstance(item, dict):
                    out.update(flatten_record(item, f"{fk}.{i}"))
                elif isinstance(item, bool):
                    out[f"{fk}.{i}"] = float(item)
                elif isinstance(item, (int, float)):
                    val = float(item)
                    out[f"{fk}.{i}"] = 0.0 if not math.isfinite(val) else val
                # NOTE: strings in lists are SKIPPED - matching training behavior
        elif isinstance(v, bool):
            out[fk] = float(v)
        elif isinstance(v, (int, float)):
            val = float(v)
            out[fk] = 0.0 if not math.isfinite(val) else val
    return out

# ------------------------------------------------------------------------------
def load_json_record(raw_text):
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("Empty file")

    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
    except json.JSONDecodeError:
        pass

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue

    try:
        data = json.loads(raw_text)
        if isinstance(data, str):
            inner = json.loads(data)
            if isinstance(inner, dict):
                return inner
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        data = json.loads(raw_text)
        if isinstance(data, dict) and len(data) == 1:
            key = list(data.keys())[0]
            if key.startswith("{") or key.startswith("["):
                inner = json.loads(key)
                if isinstance(inner, dict):
                    return inner
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        data = json.loads(raw_text)
        if isinstance(data, dict) and len(data) == 1:
            key = list(data.keys())[0]
            value = list(data.values())[0]
            if key.startswith("{") and isinstance(value, str):
                reconstructed = key + ": " + value
                try:
                    inner = json.loads(reconstructed)
                    if isinstance(inner, dict):
                        return inner
                except json.JSONDecodeError:
                    try:
                        inner = json.loads(reconstructed + "}")
                        if isinstance(inner, dict):
                            return inner
                    except json.JSONDecodeError:
                        pass
    except (json.JSONDecodeError, TypeError):
        pass

    if raw_text.startswith('"') and raw_text.endswith('"'):
        try:
            unquoted = json.loads(raw_text)
            if isinstance(unquoted, str):
                inner = json.loads(unquoted)
                if isinstance(inner, dict):
                    return inner
        except (json.JSONDecodeError, TypeError):
            pass

    if "\\\"" in raw_text:
        try:
            import ast
            unescaped = ast.literal_eval(f'"{raw_text}"')
            inner = json.loads(unescaped)
            if isinstance(inner, dict):
                return inner
        except (ValueError, SyntaxError, json.JSONDecodeError):
            pass

    raise ValueError(f"Could not parse JSON. Preview: {repr(raw_text[:300])}")

# ------------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: extract_features_fixed.py <pe_file_or_json>"}))
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(json.dumps({"error": f"File not found: {input_path}"}))
        sys.exit(1)

    try:
        with open(FEATURE_COLS_PATH, "rb") as f:
            feature_cols = pickle.load(f)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load feature_columns.pkl: {e}"}))
        sys.exit(1)

    if input_path.suffix.lower() == ".json":
        raw = input_path.read_text(encoding="utf-8")
        try:
            record = load_json_record(raw)
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.exit(1)
        if not isinstance(record, dict):
            print(json.dumps({"error": f"Expected dict, got {type(record).__name__}"}))
            sys.exit(1)
    else:
        from extract_corrected import extract_all_features
        record = extract_all_features(input_path)

    flat = flatten_record(record)
    features = np.zeros(len(feature_cols), dtype=np.float32)
    missing_cols = []

    for i, col in enumerate(feature_cols):
        val = flat.get(col)
        if val is None:
            missing_cols.append(col)
            features[i] = 0.0
        elif not np.isfinite(val):
            features[i] = 0.0
        else:
            features[i] = float(val)

    n_present = len(feature_cols) - len(missing_cols)
    print(f"Features present: {n_present}/{len(feature_cols)} "
          f"({100*n_present/len(feature_cols):.1f}%)", file=sys.stderr)
    print(f"Non-zero features: {np.count_nonzero(features)}/{len(features)}", file=sys.stderr)
    if missing_cols:
        print(f"Missing features ({len(missing_cols)}): "
              f"{missing_cols[:10]}{' ...' if len(missing_cols) > 10 else ''}", file=sys.stderr)
    
    # Debug: show what imports look like with this flattener
    import_keys = [k for k in flat.keys() if k.startswith("imports.")]
    print(f"Import keys found: {len(import_keys)}", file=sys.stderr)
    if import_keys:
        print(f"First 5 import keys: {import_keys[:5]}", file=sys.stderr)
    
    for check in ["section.sections.0.entropy",
                  "byteentropy.0",
                  "strings.numstrings"]:
        print(f"  {check}: {flat.get(check, 'MISSING')}", file=sys.stderr)
    
    if flat:
        print(f"Sample keys extracted: {sorted(flat.keys())[:8]}", file=sys.stderr)
    else:
        print(f"WARNING: flat dict is empty. Record keys: {list(record.keys())}", file=sys.stderr)

    print(json.dumps(features.tolist()))

if __name__ == "__main__":
    main()


