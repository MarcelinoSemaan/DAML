#!/usr/bin/env python3
r"""
Diagnostic: compare feature_columns.pkl against any JSON/PE sample.
Usage:
    python diagnose.py "C:\Users\Dell\Downloads\Malware1.json"
"""
import sys
import json
import pickle
import math
from pathlib import Path

FEATURE_COLS_PATH = Path(
    r"C:\Users\Dell\Downloads\DAML-LSTM-Scaler-version\DAML-LSTM-Scaler-version\AI Model\feature_columns.pkl"
)

METADATA_KEYS = {"sha256", "md5", "appeared", "label", "avclass",
                 "feature_version", "subset", "source"}
MAX_LIST_LEN = 256


def flatten_record(record, prefix=""):
    out = {}
    for k, v in record.items():
        fk = f"{prefix}.{k}" if prefix else k
        if fk in METADATA_KEYS:
            continue
        if isinstance(v, dict):
            is_imports = (
                v and
                all(isinstance(val, list) for val in v.values()) and
                any(len(val) > 0 and isinstance(val[0], str)
                    for val in v.values() if len(val) > 0)
            )
            if is_imports:
                for dll_name, funcs in v.items():
                    dll_key = dll_name.lower()
                    for func in funcs[:MAX_LIST_LEN]:
                        if isinstance(func, str):
                            out[f"{fk}.{dll_key}.{func}"] = 1.0
                        elif isinstance(func, (int, float)):
                            val = float(func)
                            out[f"{fk}.{dll_key}.{int(func)}"] = 0.0 if not math.isfinite(val) else val
            else:
                out.update(flatten_record(v, fk))
        elif isinstance(v, list):
            for i, item in enumerate(v[:MAX_LIST_LEN]):
                if isinstance(item, dict):
                    out.update(flatten_record(item, f"{fk}.{i}"))
                elif isinstance(item, str):
                    pass
                elif isinstance(item, bool):
                    out[f"{fk}.{i}"] = float(item)
                elif isinstance(item, (int, float)):
                    val = float(item)
                    out[f"{fk}.{i}"] = 0.0 if not math.isfinite(val) else val
        elif isinstance(v, bool):
            out[fk] = float(v)
        elif isinstance(v, (int, float)):
            val = float(v)
            out[fk] = 0.0 if not math.isfinite(val) else val
        elif isinstance(v, str):
            pass
    return out


def load_json_record(raw_text):
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("Empty file")

    # 1) Normal JSON object or array
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
    except json.JSONDecodeError:
        pass

    # 2) JSONL - first valid line
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

    # 3) Double-encoded string: "\"{...}\""
    try:
        data = json.loads(raw_text)
        if isinstance(data, str):
            inner = json.loads(data)
            if isinstance(inner, dict):
                return inner
    except (json.JSONDecodeError, TypeError):
        pass

    # 4) Single-key wrapper where the key IS the JSON content
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

    # 5) SPLIT double-encoding: key is partial JSON, value is the rest
    #    e.g. {"{\"md5\"": "\"abc\", \"sha1\": ..."}
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict) and len(data) == 1:
            key = list(data.keys())[0]
            value = list(data.values())[0]
            if key.startswith("{") and isinstance(value, str):
                # Reconstruct: key + ": " + value + "}"
                # The key starts with { but may not end with }
                # The value continues the JSON content
                reconstructed = key + ": " + value
                # Try adding closing brace if needed
                try:
                    inner = json.loads(reconstructed)
                    if isinstance(inner, dict):
                        return inner
                except json.JSONDecodeError:
                    # Try with closing brace
                    try:
                        inner = json.loads(reconstructed + "}")
                        if isinstance(inner, dict):
                            return inner
                    except json.JSONDecodeError:
                        pass
    except (json.JSONDecodeError, TypeError):
        pass

    # 6) The ENTIRE file is a JSON string literal (escaped quotes)
    if raw_text.startswith('"') and raw_text.endswith('"'):
        try:
            unquoted = json.loads(raw_text)
            if isinstance(unquoted, str):
                inner = json.loads(unquoted)
                if isinstance(inner, dict):
                    return inner
        except (json.JSONDecodeError, TypeError):
            pass

    # 7) Raw string with escaped quotes but not wrapped in outer quotes
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


def main():
    if len(sys.argv) < 2:
        print("Usage: diagnose.py <file.json|file.exe>")
        sys.exit(1)

    p = Path(sys.argv[1])
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)

    with open(FEATURE_COLS_PATH, "rb") as f:
        feature_cols = pickle.load(f)
    expected = set(feature_cols)
    print(f"feature_columns.pkl loaded: {len(feature_cols)} expected keys")
    print(f"  First 10 expected: {feature_cols[:10]}")
    print(f"  Last  10 expected: {feature_cols[-10:]}")
    print()

    raw = p.read_text(encoding="utf-8")
    print(f"Raw file preview (first 200 chars): {repr(raw[:200])}")
    print()

    record = load_json_record(raw)
    print(f"Parsed top-level keys: {sorted(record.keys())}")
    print()

    flat = flatten_record(record)
    actual = set(flat.keys())
    print(f"Flattened keys produced: {len(actual)}")
    print(f"  Sample produced: {sorted(actual)[:15]}")
    print()

    missing = expected - actual
    extra = actual - expected
    match = expected & actual
    print(f"MATCH:   {len(match)}")
    print(f"MISSING: {len(missing)}")
    print(f"EXTRA:   {len(extra)}")
    print()

    if missing:
        print("Missing samples:")
        for m in sorted(missing)[:20]:
            print(f"  - {m}")
    if extra:
        print("Extra samples:")
        for e in sorted(extra)[:20]:
            print(f"  + {e}")


if __name__ == "__main__":
    main()