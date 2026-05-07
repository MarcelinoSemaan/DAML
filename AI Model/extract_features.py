"""
extract_features.py
====================
Extracts features from a PE (or non-PE) file in the exact format expected
by the EmberLSTM model trained in daml.py on the EMBER2024 dataset.

Pipeline (mirrors daml.py exactly):
  1. thrember.PEFeatureExtractor().raw_features(bytez)
       → same JSON dict the JSONL training files contain
  2. flatten_record()           ← verbatim copy from daml.py
       → flat {key: float} dict
  3. Align to saved feat_cols schema  (zero-fill unknowns)
  4. StandardScaler.transform()
  5. Pad → (N_TIMESTEPS × TARGET_NF) and reshape for the LSTM

Installation
------------
    pip install pefile scikit-learn torch numpy tqdm
    pip install git+https://github.com/FutureComputing4AI/EMBER2024.git

    # signify / authenticode (optional but recommended):
    pip install signify
    # On Ubuntu 22+/24 you may also need the patched oscrypto:
    pip install "oscrypto @ git+https://github.com/wbond/oscrypto.git"

Required artefacts  (produced by daml.py training run)
-------------------------------------------------------
    <memmap_dir>/feat_cols.pkl   <- sorted column list (generate with
                                   --save-feat-cols if missing)
    <memmap_dir>/scaler.pkl      <- fitted StandardScaler

Usage
-----
    # Generate feat_cols.pkl once from training data
    python extract_features.py --save-feat-cols \\
        --data-dir /path/to/Combined-DB --memmap-dir ./memmap_cache

    # Inspect the raw thrember JSON for a file (no artefacts needed)
    python extract_features.py sample.exe --show-record

    # Get feature vector shape (no model needed)
    python extract_features.py sample.exe --memmap-dir ./memmap_cache

    # Run full inference
    python extract_features.py sample.exe \\
        --memmap-dir ./memmap_cache --model ember_lstm_best.pt

    # Batch
    python extract_features.py *.exe \\
        --memmap-dir ./memmap_cache --model ember_lstm_best.pt
"""

import argparse
import json
import math
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Graceful signify/oscrypto fallback
# signify uses oscrypto which fails on Ubuntu 22+/24 with OpenSSL 3.x.
# We stub it out so thrember can import cleanly. Authenticode features will
# be zeroed for unsigned files, which is the correct behaviour anyway.
# ─────────────────────────────────────────────────────────────────────────────
# ── signify stub — MUST be installed before thrember is touched ──────────────
def _mock_signify():
    """Inject a minimal signify stub so thrember imports without errors."""
    m_sig  = types.ModuleType("signify")
    m_auth = types.ModuleType("signify.authenticode")
    m_exc  = types.ModuleType("signify.exceptions")

    class _FakeErr(Exception):
        pass

    m_exc.SignerInfoParseError = _FakeErr
    m_exc.ParseError           = _FakeErr

    class _FakeSPEF:
        def __init__(self, *a, **kw): pass
        def iter_signed_datas(self): return iter([])

    m_auth.SignedPEFile = _FakeSPEF
    m_sig.authenticode  = m_auth
    m_sig.exceptions    = m_exc

    for name, mod in [
        ("signify",              m_sig),
        ("signify.authenticode", m_auth),
        ("signify.exceptions",   m_exc),
        ("signify.pkcs7",        types.ModuleType("signify.pkcs7")),
        ("signify.x509",         types.ModuleType("signify.x509")),
    ]:
        sys.modules[name] = mod


# 1. Always install stub first so thrember never sees a broken import
_mock_signify()

# 2. Try to replace stub with real signify only if SignedPEFile actually exists
try:
    import importlib
    for _m in list(sys.modules):          # clear our stubs
        if _m.startswith("signify"):
            del sys.modules[_m]
    import signify.authenticode           # noqa: F401
    _ = signify.authenticode.SignedPEFile # verify the symbol exists
except Exception:
    _mock_signify()                       # re-install stub
    print("[WARN] signify incompatible — authenticode features zeroed.",
          file=sys.stderr)

from thrember.features import PEFeatureExtractor   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match daml.py exactly
# ─────────────────────────────────────────────────────────────────────────────
TARGET_NF     = 64
MAX_LIST_LEN  = 256
METADATA_KEYS = {
    'sha256', 'md5', 'appeared', 'label', 'avclass',
    'feature_version', 'subset', 'source',
}


# ─────────────────────────────────────────────────────────────────────────────
# flatten_record — verbatim copy from daml.py  (DO NOT MODIFY)
# ─────────────────────────────────────────────────────────────────────────────
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
                    out[f"{full_key}.{i}"] = 0.0 if not np.isfinite(val) else val
        elif isinstance(v, bool):
            out[full_key] = float(v)
        elif isinstance(v, (int, float)):
            val = float(v)
            out[full_key] = 0.0 if not np.isfinite(val) else val
    return out


# ─────────────────────────────────────────────────────────────────────────────
# EmberLSTM — verbatim copy from daml.py  (needed for inference)
# ─────────────────────────────────────────────────────────────────────────────
class EmberLSTM(nn.Module):
    def __init__(self, n_features: int, n_timesteps: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            128, 256, num_layers=2,
            batch_first=True, dropout=0.3, bidirectional=True,
        )
        self.attn       = nn.Linear(512, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x           = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        weights     = torch.softmax(
            self.attn(lstm_out).squeeze(-1), dim=1
        ).unsqueeze(-1)
        context = (lstm_out * weights).sum(dim=1)
        return self.classifier(context).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Artefact loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_artifacts(memmap_dir: str):
    """
    Load feat_cols.pkl and scaler.pkl that daml.py produced during training.
    """
    mdir    = Path(memmap_dir)
    fc_path = mdir / 'feat_cols.pkl'
    sc_path = mdir / 'scaler.pkl'

    if not fc_path.exists():
        sys.exit(
            f"\n[ERROR] {fc_path} not found.\n"
            "Generate it once with:\n\n"
            f"    python extract_features.py --save-feat-cols "
            f"--data-dir /path/to/Combined-DB --memmap-dir {memmap_dir}\n"
        )
    if not sc_path.exists():
        sys.exit(f"\n[ERROR] {sc_path} not found. Run daml.py training first.")

    with open(fc_path, 'rb') as f:
        feat_cols = pickle.load(f)
    with open(sc_path, 'rb') as f:
        scaler = pickle.load(f)

    return feat_cols, scaler


def load_model(
    model_path: str,
    n_cols: int,
    target_nf: int    = TARGET_NF,
    device: torch.device = torch.device('cpu'),
) -> EmberLSTM:
    n_timesteps = max(1, math.ceil(n_cols / target_nf))
    model       = EmberLSTM(target_nf, n_timesteps).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction pipeline
# ─────────────────────────────────────────────────────────────────────────────

_extractor = PEFeatureExtractor()   # single instance, reused across calls


def file_to_raw_record(filepath: str) -> dict:
    """
    Step 1 — bytes → thrember raw JSON record.
    Produces the same dict structure as each JSONL line in EMBER2024.
    Keys: sha256, general, histogram, byteentropy, strings, header,
          section, imports, exports, datadirectories, richheader,
          authenticode, pefilewarnings
    """
    bytez = Path(filepath).read_bytes()
    return _extractor.raw_features(bytez)


def raw_record_to_flat(record: dict) -> dict:
    """
    Step 2 — flatten_record() identical to daml.py.
    Recursively expands nested dicts/lists into dot-separated float keys.
    Metadata keys (sha256, label, …) are silently dropped.
    """
    return flatten_record(record)


def flat_to_array(flat: dict, feat_cols: list, scaler) -> np.ndarray:
    """
    Steps 3-5 — align → scale → pad → reshape.

    Returns float32 ndarray of shape (N_TIMESTEPS, TARGET_NF).
    Drop directly into the LSTM: model(tensor.unsqueeze(0))
    """
    n_cols    = len(feat_cols)
    col_index = {c: i for i, c in enumerate(feat_cols)}

    # 3. Align to training schema — zero-fill any column not seen at inference
    x_raw = np.zeros(n_cols, dtype=np.float32)
    for col, val in flat.items():
        idx = col_index.get(col)
        if idx is not None:
            x_raw[idx] = float(val)

    # 4. Apply the fitted StandardScaler from training
    x_scaled = scaler.transform(x_raw.reshape(1, -1))[0].astype(np.float32)

    # 5. Pad + reshape — mirrors MemmapDataset.__getitem__ exactly
    n_timesteps = max(1, math.ceil(n_cols / TARGET_NF))
    pad_to      = n_timesteps * TARGET_NF
    if len(x_scaled) < pad_to:
        x_scaled = np.pad(x_scaled, (0, pad_to - len(x_scaled)),
                          constant_values=0.0)

    return x_scaled.reshape(n_timesteps, TARGET_NF)   # (T, 64)


def extract_tensor(
    filepath: str,
    feat_cols: list,
    scaler,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Full pipeline: file path → (1, N_TIMESTEPS, TARGET_NF) torch.Tensor.
    Pass directly to model() for inference.
    """
    record = file_to_raw_record(filepath)
    flat   = raw_record_to_flat(record)
    arr    = flat_to_array(flat, feat_cols, scaler)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# feat_cols.pkl generator — run once after training
# ─────────────────────────────────────────────────────────────────────────────
def save_feat_cols(data_dir: str, memmap_dir: str) -> list:
    """
    Reproduces Pass-1 (discover_schema) from daml.py and writes feat_cols.pkl.
    Only needed once per dataset / training run.
    """
    import json as _json

    files       = sorted(Path(data_dir).glob("*.jsonl"))
    train_files = [p for p in files if 'train' in p.name.lower()]
    if not train_files:
        sys.exit(f"[ERROR] No *train*.jsonl files found in {data_dir}")

    feat_cols = None
    print(f"Scanning {len(train_files)} training file(s) for schema …")
    for p in train_files:
        with p.open('r', encoding='utf-8') as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = _json.loads(raw)
                    if int(rec.get('label', -1)) >= 0:
                        feat_cols = sorted(flatten_record(rec).keys())
                        break
                except Exception:
                    continue
        if feat_cols is not None:
            break

    if feat_cols is None:
        sys.exit("[ERROR] No labelled records found. Check --data-dir path.")

    out = Path(memmap_dir) / 'feat_cols.pkl'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(feat_cols, f)
    print(f"Saved {len(feat_cols)} feature columns → {out}")
    return feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "EMBER2024 feature extractor — inference pipeline for EmberLSTM\n"
            "Uses thrember (pefile-based) to produce the same JSON as the\n"
            "EMBER2024 JSONL training files, then applies the daml.py pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('files', nargs='*',
                        help='PE/binary file(s) to analyse')
    parser.add_argument('--memmap-dir', default='memmap_cache',
                        help='Directory with feat_cols.pkl + scaler.pkl '
                             '(default: ./memmap_cache)')
    parser.add_argument('--model', default=None,
                        help='Path to ember_lstm_best.pt  (omit to only show '
                             'feature vector dimensions)')
    parser.add_argument('--save-feat-cols', action='store_true',
                        help='Generate feat_cols.pkl from training JSONL and exit')
    parser.add_argument('--data-dir', default=None,
                        help='EMBER2024 Combined-DB directory '
                             '(required with --save-feat-cols)')
    parser.add_argument('--show-record', action='store_true',
                        help='Print the raw thrember JSON record '
                             '(no artefacts needed)')
    args = parser.parse_args()

    # ── Generate feat_cols.pkl ──────────────────────────────────────────────
    if args.save_feat_cols:
        if not args.data_dir:
            sys.exit("[ERROR] --data-dir is required with --save-feat-cols")
        save_feat_cols(args.data_dir, args.memmap_dir)
        return

    if not args.files:
        parser.print_help()
        return

    # ── --show-record: no artefacts needed ──────────────────────────────────
    if args.show_record:
        for filepath in args.files:
            fp = Path(filepath)
            if not fp.exists():
                print(f"[SKIP] {fp} — not found")
                continue
            record = file_to_raw_record(str(fp))
            print(f"\n{'─'*60}\nFile: {fp}")
            print(json.dumps(record, indent=2, default=str))
        return

    # ── Load artefacts ──────────────────────────────────────────────────────
    feat_cols, scaler = load_artifacts(args.memmap_dir)
    n_cols      = len(feat_cols)
    n_timesteps = max(1, math.ceil(n_cols / TARGET_NF))
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Schema : {n_cols} features  →  LSTM input {n_timesteps} × {TARGET_NF}")
    print(f"Device : {device}")

    # ── Load model (optional) ───────────────────────────────────────────────
    model = None
    if args.model:
        model = load_model(args.model, n_cols, TARGET_NF, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model  : {args.model}  ({n_params:,} params)")

    # ── Process files ───────────────────────────────────────────────────────
    print()
    hdr = f"{'File':<50}  {'Result'}"
    print(hdr); print('─' * 80)

    for filepath in args.files:
        fp = Path(filepath)
        if not fp.exists():
            print(f"{'[NOT FOUND] ' + str(fp):<50}")
            continue
        try:
            tensor = extract_tensor(str(fp), feat_cols, scaler, device)

            if model is not None:
                with torch.no_grad():
                    prob = torch.sigmoid(model(tensor)).item()
                tag    = "MALICIOUS" if prob >= 0.5 else "BENIGN"
                result = f"prob={prob:.4f}  [{tag}]"
            else:
                arr    = tensor.squeeze(0).cpu().numpy()
                nz     = int((arr != 0).sum())
                result = f"shape={arr.shape}  non-zero={nz}/{arr.size}"

            print(f"{fp.name:<50}  {result}")

        except Exception as exc:
            print(f"{fp.name:<50}  [ERROR] {exc}")


if __name__ == '__main__':
    main()