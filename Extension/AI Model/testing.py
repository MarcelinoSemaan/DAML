# testing.py
import json
import pickle
import numpy as np
from pathlib import Path

# Import the functions from your existing extractor
from extract_features import file_to_raw_record, raw_record_to_flat, flatten_record, load_artifacts

# ── Load artifacts ──────────────────────────────────────────────────────────
feat_cols, scaler = load_artifacts("./memmap_cache")

# ── Extract hello.exe features ──────────────────────────────────────────────
print("=" * 60)
print("EXTRACTING: hello64.exe")
record_hello = file_to_raw_record("hello.exe")
flat_hello = raw_record_to_flat(record_hello)

print(f"Raw keys extracted: {len(flat_hello)}")
print(f"Sample keys: {list(flat_hello.keys())[:5]}")

# ── Compare to a known benign from your dataset ─────────────────────────────
# Replace this path with an actual benign JSON from your test set
BENIGN_JSON_PATH = r"C:\Users\Dell\Downloads\malware_1_7d05c4224d514c6fc1554049084a4ade.json"

print("\n" + "=" * 60)
if not Path(BENIGN_JSON_PATH).exists():
    print(f"[WARN] Benign JSON not found at: {BENIGN_JSON_PATH}")
    print("Please update BENIGN_JSON_PATH to a real benign EMBER sample.")
else:
    with open(BENIGN_JSON_PATH, "r", encoding="utf-8") as f:
        benign_record = json.load(f)
    flat_benign = flatten_record(benign_record)
    
    print(f"COMPARING hello.exe vs {Path(BENIGN_JSON_PATH).name}")
    print(f"Benign keys extracted: {len(flat_benign)}")
    
    # Find large differences
    all_keys = set(flat_hello.keys()) | set(flat_benign.keys())
    diffs = []
    for key in all_keys:
        v1 = float(flat_hello.get(key, 0))
        v2 = float(flat_benign.get(key, 0))
        if abs(v1 - v2) > 50:  # threshold for "significant"
            diffs.append((key, v1, v2, abs(v1 - v2)))
    
    diffs.sort(key=lambda x: x[3], reverse=True)
    print(f"\nTop 10 feature differences (|diff| > 50):")
    print(f"{'Feature':<50} {'hello.exe':>12} {'benign':>12} {'|diff|':>12}")
    print("-" * 90)
    for key, v1, v2, d in diffs[:10]:
        print(f"{key:<50} {v1:>12.1f} {v2:>12.1f} {d:>12.1f}")

# ── Check scaled output ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SCALED FEATURE STATISTICS")

n_cols = len(feat_cols)
col_index = {c: i for i, c in enumerate(feat_cols)}

x_raw = np.zeros(n_cols, dtype=np.float32)
for col, val in flat_hello.items():
    idx = col_index.get(col)
    if idx is not None:
        x_raw[idx] = float(val)

x_scaled = scaler.transform(x_raw.reshape(1, -1))[0].astype(np.float32)

print(f"Min scaled value : {x_scaled.min():.4f}")
print(f"Max scaled value : {x_scaled.max():.4f}")
print(f"Mean scaled value: {x_scaled.mean():.4f}")
print(f"Std scaled value : {x_scaled.std():.4f}")
print(f"Values > |3.0|   : {(np.abs(x_scaled) > 3.0).sum()} of {x_scaled.size}")
print(f"Values > |5.0|   : {(np.abs(x_scaled) > 5.0).sum()} of {x_scaled.size}")
print(f"Values > |10.0|  : {(np.abs(x_scaled) > 10.0).sum()} of {x_scaled.size}")

if (np.abs(x_scaled) > 5.0).sum() > 20:
    print("\n[DIAGNOSIS] Many extreme scaled values detected.")
    print("            This confirms scaling drift / out-of-distribution input.")
else:
    print("\n[DIAGNOSIS] Scaled values look mostly normal.")