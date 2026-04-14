from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import math
from typing import List

app = FastAPI(title="DAML API")

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MATCH YOUR TRAINED MODEL (64 features, not 107!)
N_FEATURES = 64
N_TIMESTEPS = 40    # 40 * 64 = 2560
PAD_TO = 2560       # Model was trained with padding to 2560

# EMBER returns 2381 or 2568, we'll pad/truncate to 2560
N_COLS_RAW = 2568   # What EMBER extract_features.py returns

print(f"DAML API on {DEVICE}")
print(f"Raw features: {N_COLS_RAW} → pad to {PAD_TO} ({N_TIMESTEPS}×{N_FEATURES})")

# ── Model (matches your trained model with 64 features) ─────────────────────
class EmberLSTM(nn.Module):
    def __init__(self, n_features: int, n_timesteps: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 128),    # [128, 64] - matches your checkpoint
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
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model error: {e}")
    raise

# ── API ───────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float]

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "device": str(DEVICE),
        "model": "EmberLSTM",
        "config": {
            "n_features": N_FEATURES,
            "n_timesteps": N_TIMESTEPS,
            "pad_to": PAD_TO,
            "input_raw": N_COLS_RAW
        }
    }

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        # Accept 2568 features from EMBER extractor
        if len(request.features) != N_COLS_RAW:
            raise ValueError(f"Expected {N_COLS_RAW} features, got {len(request.features)}")
        
        # Preprocess: truncate from 2568 to 2560 (remove last 8 features)
        x = np.array(request.features, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Truncate to match model's expected input size
        x = x[:PAD_TO]
        
        # Reshape to (1, 40, 64)
        x = x.reshape(1, N_TIMESTEPS, N_FEATURES)
        x = torch.tensor(x).to(DEVICE)
        
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