from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from typing import List
from pathlib import Path

app = FastAPI(title="DAML API")

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MATCH CHECKPOINT: 64 features, 24 timesteps = 1536 total
N_FEATURES = 64
N_TIMESTEPS = 24
EXPECTED_FEATURES = N_TIMESTEPS * N_FEATURES  # 1536

# ── Path Setup ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "ember_lstm_best.pt"

print(f"DAML API on {DEVICE}")
print(f"Looking for model at: {MODEL_PATH.absolute()}")
print(f"Model expects: {N_TIMESTEPS}×{N_FEATURES} = {EXPECTED_FEATURES}")

# ── Model (matches checkpoint) ───────────────────────────────────────────
class EmberLSTM(nn.Module):
    def __init__(self, n_features: int, n_timesteps: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 128),  # 64→128
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
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded successfully")
    print(f"   Input layer: Linear(in={N_FEATURES}, out=128)")
except Exception as e:
    print(f"❌ Model error: {e}")

# ── API Endpoints ─────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float]  # Now expects exactly 1536 features

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail=f"Model not found at {MODEL_PATH}")
        
        # Now expects exactly 1536 features (64 × 24)
        if len(request.features) != EXPECTED_FEATURES:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {EXPECTED_FEATURES} features (64×24), got {len(request.features)}"
            )
        
        # Preprocess: reshape to (1, 24, 64)
        x = np.array(request.features, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reshape for LSTM: (batch=1, timesteps=24, features=64)
        x = x.reshape(1, N_TIMESTEPS, N_FEATURES)
        x = torch.tensor(x).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                prob = torch.sigmoid(model(x)).item()
        
        # Confidence levels
        if prob > 0.9: conf = "critical"
        elif prob > 0.7: conf = "high"
        elif prob > 0.5: conf = "medium"
        else: conf = "low"
        
        return {
            "is_malicious": prob > 0.5,
            "confidence": conf,
            "malicious_probability": round(prob, 4),
            "model_dims": f"{N_TIMESTEPS}×{N_FEATURES}",
            "note": f"Using {EXPECTED_FEATURES} features (selected 64 per timestep)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)