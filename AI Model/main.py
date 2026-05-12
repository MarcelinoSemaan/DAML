"""
DAML FastAPI Server
===================
Run from inside "AI Model":
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import json
import math
import pickle
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Graceful signify fallback ───────────────────────────────────────────────
def _mock_signify():
    m_sig = types.ModuleType("signify")
    m_auth = types.ModuleType("signify.authenticode")
    m_exc = types.ModuleType("signify.exceptions")

    class _FakeErr(Exception):
        pass

    m_exc.SignerInfoParseError = _FakeErr
    m_exc.ParseError = _FakeErr

    class _FakeSPEF:
        def __init__(self, *a, **kw):
            pass
        def iter_signed_datas(self):
            return iter([])

    m_auth.SignedPEFile = _FakeSPEF
    m_sig.authenticode = m_auth
    m_sig.exceptions = m_exc

    for name, mod in [
        ("signify", m_sig),
        ("signify.authenticode", m_auth),
        ("signify.exceptions", m_exc),
        ("signify.pkcs7", types.ModuleType("signify.pkcs7")),
        ("signify.x509", types.ModuleType("signify.x509")),
    ]:
        sys.modules[name] = mod


try:
    from signify.authenticode import SignedPEFile
    import signify.exceptions
except Exception:
    print("[WARN] signify/oscrypto not available — authenticode features will be zero.", file=sys.stderr)
    _mock_signify()

from thrember.features import PEFeatureExtractor

# ── Config ──────────────────────────────────────────────────────────────────
TARGET_NF = 64
MAX_LIST_LEN = 256
METADATA_KEYS = {
    "sha256", "md5", "appeared", "label", "avclass",
    "feature_version", "subset", "source",
}

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "ember_lstm_best.pt"
FEAT_COLS_PATH = ARTIFACTS_DIR / "feat_cols.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"

# ── Model ───────────────────────────────────────────────────────────────────
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
        self.attn = nn.Linear(512, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1).unsqueeze(-1)
        context = (lstm_out * weights).sum(dim=1)
        return self.classifier(context).squeeze(-1)


# ── Feature extraction ──────────────────────────────────────────────────────
_extractor = PEFeatureExtractor()


def flatten_record(record: dict, prefix: str = "") -> dict:
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

try:
    with open("artifacts/meta_indices.json") as f:
        META_INDICES = json.load(f)["indices"]
except:
    META_INDICES = []

def record_to_features(record: dict, feat_cols: list, scaler):
    """Raw EMBER record dict → (N_TIMESTEPS, TARGET_NF) array."""
    flat = flatten_record(record)
    n_cols = len(feat_cols)
    col_index = {c: i for i, c in enumerate(feat_cols)}

    x_raw = np.zeros(n_cols, dtype=np.float32)
    for col, val in flat.items():
        idx = col_index.get(col)
        if idx is not None:
            x_raw[idx] = float(val)

    # ── FAST FIX: neutralize metadata columns ──
    for idx in META_INDICES:
        if x_raw[idx] == 0.0:
            x_raw[idx] = scaler.mean_[idx]

    x_scaled = scaler.transform(x_raw.reshape(1, -1))[0].astype(np.float32)
    n_timesteps = max(1, math.ceil(n_cols / TARGET_NF))
    pad_to = n_timesteps * TARGET_NF
    if len(x_scaled) < pad_to:
        x_scaled = np.pad(x_scaled, (0, pad_to - len(x_scaled)), constant_values=0.0)

    return x_scaled.reshape(n_timesteps, TARGET_NF)


def file_to_features(file_path: str, feat_cols: list, scaler):
    bytez = Path(file_path).read_bytes()
    record = _extractor.raw_features(bytez)
    return record_to_features(record, feat_cols, scaler)


# ── Explainability ──────────────────────────────────────────────────────────

def group_by_prefix(feat_name):
    """Map EMBER feature name to semantic group."""
    if feat_name.startswith("byteentropy."):
        return "byte_entropy"
    elif feat_name.startswith("histogram."):
        return "histogram"
    elif feat_name.startswith("section.") or feat_name.startswith("sections."):
        return "sections"
    elif feat_name.startswith("import.") or feat_name.startswith("imports."):
        return "imports"
    elif feat_name.startswith("export.") or feat_name.startswith("exports."):
        return "exports"
    elif feat_name.startswith("general."):
        return "general"
    elif feat_name.startswith("header.") or feat_name.startswith("optional_header."):
        return "header"
    elif feat_name.startswith("strings."):
        return "strings"
    elif feat_name.startswith("datadirectories."):
        return "data_directories"
    elif feat_name.startswith("authenticode.") or feat_name.startswith("signature."):
        return "authenticode"
    else:
        return "other"


def get_remediation_advice(feature_name: str):
    """Return human-readable remediation for common malicious EMBER features."""
    prefix = feature_name.split('.')[0] if '.' in feature_name else feature_name
    
    advice_map = {
        "byteentropy": "Unpack the binary or avoid high-entropy packers (UPX, VMProtect).",
        "histogram": "Avoid polymorphic/obfuscated byte patterns that distort the histogram.",
        "section": "Remove packed sections and avoid RWX memory permissions.",
        "sections": "Remove packed sections and avoid RWX memory permissions.",
        "import": "Remove suspicious APIs: VirtualAlloc, WriteProcessMemory, CreateRemoteThread, InternetOpen.",
        "imports": "Remove suspicious APIs: VirtualAlloc, WriteProcessMemory, CreateRemoteThread, InternetOpen.",
        "export": "Add normal exports or remove obfuscated export tables.",
        "exports": "Add normal exports or remove obfuscated export tables.",
        "general": "Use normal file size and standard number of sections.",
        "header": "Fix PE header anomalies (entry point, subsystem, checksum).",
        "optional_header": "Fix optional header anomalies (image base, subsystem).",
        "strings": "Strip suspicious strings: URLs, IPs, registry paths, cmd.exe, PowerShell.",
        "datadirectories": "Restore standard data directory layout.",
        "authenticode": "Sign the executable with a valid Authenticode certificate.",
        "signature": "Sign the executable with a valid Authenticode certificate.",
    }
    return advice_map.get(prefix)


def compute_attribution(model, tensor, feat_cols, n_cols, n_timesteps):
    """
    Compute feature attributions using gradient-based method.
    Returns top features and group contributions.
    """
    baseline = torch.zeros_like(tensor)

    def forward_raw(x):
        proj = model.input_proj(x)
        lstm_out, _ = model.lstm(proj)
        attn_scores = model.attn(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (lstm_out * attn_weights).sum(dim=1)
        return model.classifier(context).squeeze(-1)

    # Simple gradient attribution (faster than IG, good enough for UI)
    tensor.requires_grad = True
    output = forward_raw(tensor)
    prob = float(torch.sigmoid(output).item())
    output.backward()

    grad_np = tensor.grad[0].cpu().numpy().flatten()[:n_cols]
    # Convert to native Python floats so Pydantic/JSON can serialize the response
    attr_np = grad_np.astype(float).tolist()
    attr_abs = [abs(x) for x in attr_np]

    # Top 15 features
    top_k = 15
    top_idx = np.argsort(attr_abs)[-top_k:][::-1]
    top_features = []
    for rank, idx in enumerate(top_idx, 1):
        feat = feat_cols[idx]
        val = attr_abs[idx]
        signed = attr_np[idx]
        direction = "malicious" if signed > 0 else "benign"
        top_features.append({
            "rank": rank,
            "feature": feat,
            "group": group_by_prefix(feat),
            "attribution": val,
            "direction": direction,
            "raw_value": signed
        })

    # Group contributions
    group_scores = {}
    group_directional = {}
    for i, feat in enumerate(feat_cols):
        g = group_by_prefix(feat)
        group_scores[g] = group_scores.get(g, 0.0) + attr_abs[i]
        group_directional[g] = group_directional.get(g, 0.0) + attr_np[i]

    total = sum(group_scores.values())
    groups = []
    for g, score in sorted(group_scores.items(), key=lambda x: -x[1]):
        pct = score / total * 100 if total > 0 else 0.0
        net_dir = group_directional[g]
        groups.append({
            "group": g,
            "percentage": round(pct, 1),
            "direction": "malicious" if net_dir > 0 else "benign",
            "score": score
        })

    # Benign pushers: features with negative attribution (push toward benign)
    benign_pushers = []
    for i, feat in enumerate(feat_cols):
        if attr_np[i] < -0.001:
            benign_pushers.append({
                "feature": feat,
                "attribution": attr_np[i],
                "group": group_by_prefix(feat)
            })
    benign_pushers.sort(key=lambda x: x["attribution"])
    benign_pushers = benign_pushers[:10]

    # Generate actionable remediation for top malicious features
    recommendations = []
    seen_groups = set()
    for f in top_features:
        if f["direction"] == "malicious":
            advice = get_remediation_advice(f["feature"])
            if advice and f["group"] not in seen_groups:
                recommendations.append({
                    "feature": f["feature"],
                    "group": f["group"],
                    "advice": advice
                })
                seen_groups.add(f["group"])
                if len(recommendations) >= 3:
                    break

    return {
        "probability": prob,
        "prediction": "MALICIOUS" if prob >= 0.5 else "BENIGN",
        "top_features": top_features,
        "groups": groups,
        "benign_pushers": benign_pushers,
        "recommendations": recommendations
    }


# ── Lifespan / Startup ──────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.model = None
        self.feat_cols = None
        self.scaler = None
        self.n_cols = 0
        self.n_timesteps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


state = State()


def load_artifacts():
    for p, name in [(FEAT_COLS_PATH, "feat_cols.pkl"),
                    (SCALER_PATH, "scaler.pkl"),
                    (MODEL_PATH, "ember_lstm_best.pt")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found at {p}")

    with open(FEAT_COLS_PATH, "rb") as f:
        state.feat_cols = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        state.scaler = pickle.load(f)

    state.n_cols = len(state.feat_cols)
    state.n_timesteps = max(1, math.ceil(state.n_cols / TARGET_NF))

    state.model = EmberLSTM(TARGET_NF, state.n_timesteps).to(state.device)
    state.model.load_state_dict(torch.load(MODEL_PATH, map_location=state.device))
    state.model.eval()

    print(f"[OK] Model loaded: {MODEL_PATH}")
    print(f"[OK] Features: {state.n_cols} -> {state.n_timesteps}x{TARGET_NF}")
    print(f"[OK] Device: {state.device}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="DAML API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ─────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: list[float]


class PredictPathRequest(BaseModel):
    file_path: str


class PredictJsonRequest(BaseModel):
    record: dict


class PredictResponse(BaseModel):
    is_malicious: bool
    confidence: str
    malicious_probability: float


class ExplainResponse(BaseModel):
    probability: float
    prediction: str
    top_features: list[dict]
    groups: list[dict]
    benign_pushers: list[dict] = []
    recommendations: list[dict] = []


# ── Helpers ─────────────────────────────────────────────────────────────────
def _infer(tensor: torch.Tensor) -> PredictResponse:
    with torch.no_grad():
        prob = torch.sigmoid(state.model(tensor)).item()

    is_malicious = prob >= 0.5
    if prob >= 0.9:
        confidence = "high"
    elif prob >= 0.7:
        confidence = "medium"
    elif prob >= 0.5:
        confidence = "low"
    else:
        confidence = "low"

    return PredictResponse(
        is_malicious=is_malicious,
        confidence=confidence,
        malicious_probability=prob,
    )


def _prepare_array(arr: np.ndarray):
    expected = state.n_timesteps * TARGET_NF

    if arr.size == state.n_cols:
        x_scaled = state.scaler.transform(arr.reshape(1, -1))[0].astype(np.float32)
        pad_to = state.n_timesteps * TARGET_NF
        if len(x_scaled) < pad_to:
            x_scaled = np.pad(x_scaled, (0, pad_to - len(x_scaled)), constant_values=0.0)
        arr = x_scaled
    elif arr.size != expected:
        raise ValueError(f"Expected {expected} or {state.n_cols} features, got {arr.size}")

    return torch.tensor(arr.reshape(state.n_timesteps, TARGET_NF), dtype=torch.float32).unsqueeze(0).to(state.device)


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        arr = np.array(req.features, dtype=np.float32)
        tensor = _prepare_array(arr)
        return _infer(tensor)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_json", response_model=PredictResponse)
def predict_json(req: PredictJsonRequest):
    """Accept a raw EMBER JSON record directly (no file needed)."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features_arr = record_to_features(req.record, state.feat_cols, state.scaler)
        tensor = torch.tensor(features_arr, dtype=torch.float32).unsqueeze(0).to(state.device)
        return _infer(tensor)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/predict_path", response_model=PredictResponse)
def predict_path(req: PredictPathRequest):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    path = Path(req.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    try:
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                arr = np.array(data, dtype=np.float32)
                tensor = _prepare_array(arr)
            elif isinstance(data, dict) and "features" in data:
                arr = np.array(data["features"], dtype=np.float32)
                tensor = _prepare_array(arr)
            elif isinstance(data, dict):
                features_arr = record_to_features(data, state.feat_cols, state.scaler)
                tensor = torch.tensor(features_arr, dtype=torch.float32).unsqueeze(0).to(state.device)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="JSON must be a feature array, {features: [...]}, or a raw EMBER record",
                )
        else:
            features_arr = file_to_features(str(path), state.feat_cols, state.scaler)
            tensor = torch.tensor(features_arr, dtype=torch.float32).unsqueeze(0).to(state.device)

        return _infer(tensor)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/explain", response_model=ExplainResponse)
def explain(req: PredictPathRequest):
    """Explain why a file was classified as malicious or benign."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    path = Path(req.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    try:
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                arr = np.array(data, dtype=np.float32)
                x_scaled = state.scaler.transform(arr.reshape(1, -1))[0].astype(np.float32)
                pad_to = state.n_timesteps * TARGET_NF
                if len(x_scaled) < pad_to:
                    x_scaled = np.pad(x_scaled, (0, pad_to - len(x_scaled)), constant_values=0.0)
                features_arr = x_scaled.reshape(state.n_timesteps, TARGET_NF)
            elif isinstance(data, dict) and "features" not in data:
                features_arr = record_to_features(data, state.feat_cols, state.scaler)
            elif isinstance(data, dict) and "features" in data:
                arr = np.array(data["features"], dtype=np.float32)
                x_scaled = state.scaler.transform(arr.reshape(1, -1))[0].astype(np.float32)
                pad_to = state.n_timesteps * TARGET_NF
                if len(x_scaled) < pad_to:
                    x_scaled = np.pad(x_scaled, (0, pad_to - len(x_scaled)), constant_values=0.0)
                features_arr = x_scaled.reshape(state.n_timesteps, TARGET_NF)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="JSON must be a feature array, {features: [...]}, or a raw EMBER record",
                )
        else:
            features_arr = file_to_features(str(path), state.feat_cols, state.scaler)

        tensor = torch.tensor(features_arr, dtype=torch.float32).unsqueeze(0).to(state.device)
        return compute_attribution(state.model, tensor, state.feat_cols, state.n_cols, state.n_timesteps)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)