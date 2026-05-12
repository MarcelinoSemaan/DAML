import json
import pickle
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR   = '/home/marcelino/Desktop/Data/Dataset/Combined-DB'
MEMMAP_DIR = Path(DATA_DIR) / 'memmap_cache'
MEMMAP_DIR.mkdir(exist_ok=True)

METADATA_KEYS = {'sha256', 'md5', 'appeared', 'label', 'avclass',
                 'feature_version', 'subset', 'source'}
MAX_LIST_LEN  = 256

# ── FLATTENER (copied from daml.py) ──────────────────────────────────────────
def flatten_record(record: dict, prefix: str = '') -> dict:
    out = {}
    for k, v in record.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if full_key in METADATA_KEYS:
            continue
        if isinstance(v, dict):
            out.update(flatten_record(v, full_key))
        elif isinstance(v, list):
            for i, item in enumerate(v[:MAX_LIST_LEN]):
                if isinstance(item, bool):
                    out[f"{full_key}.{i}"] = float(item)
                elif isinstance(item, (int, float)):
                    val = float(item)
                    out[f"{full_key}.{i}"] = 0.0 if not (val == val and abs(val) != float('inf')) else val
        elif isinstance(v, bool):
            out[full_key] = float(v)
        elif isinstance(v, (int, float)):
            val = float(v)
            out[full_key] = 0.0 if not (val == val and abs(val) != float('inf')) else val
    return out

# ── DISCOVER ─────────────────────────────────────────────────────────────────
def discover_schema(data_dir):
    files = sorted(Path(data_dir).glob("*.jsonl"))
    print(f"Found {len(files)} JSONL files:")
    for f in files:
        print(f"  - {f.name}")

    train_files = [p for p in files if 'train' in p.name.lower()]
    print(f"Files with 'train' in name: {len(train_files)}")

    for p in train_files:
        print(f"Scanning {p.name} ...")
        with p.open('r', encoding='utf-8') as f:
            for i, raw in enumerate(f):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                    lbl = int(rec.get('label', -1))
                    if lbl >= 0:
                        cols = sorted(flatten_record(rec).keys())
                        print(f"Found labelled record at line {i+1}, {len(cols)} features")
                        return cols
                except Exception as e:
                    print(f"  Bad line {i+1}: {e}")
                    continue
    raise RuntimeError("No labelled records found for schema")

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    feat_cols = discover_schema(DATA_DIR)
    
    out_path = MEMMAP_DIR / 'feature_columns.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(feat_cols, f)
    
    print(f"\nSUCCESS: Saved {len(feat_cols)} feature columns to {out_path}")