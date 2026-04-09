from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pickle
import math
from typing import List
from sklearn.preprocessing import StandardScaler  # Added for dummy scaler

app = FastAPI(title="DAML API")

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_NF = 64

# Load feature count from training metadata (or set manually)
try:
    with open('memmap_cache/train_meta.pkl', 'rb') as f:
        N_COLS = pickle.load(f)['shape'][1]
except:
    N_COLS = 2568  # Default for thrember

N_TIMESTEPS = max(1, math.ceil(N_COLS / TARGET_NF))
N_FEATURES = TARGET_NF
PAD_TO = N_TIMESTEPS * N_FEATURES

print(f"DAML API on {DEVICE}")
print(f"Features: {N_COLS} → {PAD_TO} ({N_TIMESTEPS}×{N_FEATURES})")

# ── Model ────────────────────────────────────────────────────────────────────
class EmberLSTM(nn.Module):
    def __init__(self, n_features: int, n_timesteps: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True,
                            dropout=0.3, bidirectional=True)
        self.attn = nn.Linear(512, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1).unsqueeze(-1)
        context = (lstm_out * weights).sum(dim=1)
        return self.classifier(context).squeeze(-1)

# ── Load Model ─────────────────────────────────────────────────────────────────
model = EmberLSTM(N_FEATURES, N_TIMESTEPS).to(DEVICE)

try:
    model.load_state_dict(torch.load('ember_lstm_best.pt', map_location=DEVICE))
    model.eval()
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Model error: {e}")
    raise

# ── DUMMY SCALER (FOR TESTING ONLY) ────────────────────────────────────────────
try:
    with open('memmap_cache/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Real scaler loaded")
except FileNotFoundError:
    print("⚠️ WARNING: Using DUMMY scaler - predictions will be random!")
    print("   Get the real scaler.pkl from training for accurate results.")
    
    # Create dummy scaler (identity transform - does nothing)
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(N_COLS)
    scaler.scale_ = np.ones(N_COLS)
    scaler.n_features_in_ = N_COLS
    scaler.var_ = np.ones(N_COLS)

# ── API ───────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float]

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "device": str(DEVICE),
        "scaler_type": "dummy" if np.all(scaler.scale_ == 1) else "real"
    }

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        if len(request.features) != N_COLS:
            raise ValueError(f"Expected {N_COLS} features, got {len(request.features)}")
        
        # Preprocess
        x = np.array(request.features, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0)
        x = scaler.transform(x.reshape(1, -1))[0]
        
        if len(x) < PAD_TO:
            x = np.pad(x, (0, PAD_TO - len(x)))
        
        x = x.reshape(N_TIMESTEPS, N_FEATURES)
        x = torch.tensor(x).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        
        # Confidence levels
        if prob > 0.9: conf = "critical"
        elif prob > 0.7: conf = "high"
        elif prob > 0.5: conf = "medium"
        else: conf = "low"
        
        return {
            "is_malicious": prob > 0.5,
            "confidence": conf,
            "malicious_probability": round(prob, 4)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)