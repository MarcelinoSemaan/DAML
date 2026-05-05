from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pickle
import math
from pathlib import Path

app = FastAPI()

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR = Path(r"C:\Users\Dell\Downloads\DAML-LSTM-Scaler-version\DAML-LSTM-Scaler-version\AI Model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = DEVICE.type
TARGET_NF = 64

# ── Model (must match training exactly) ──────────────────────────────────────
class EmberLSTM(nn.Module):
    def __init__(self, n_features: int, n_timesteps: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True,
                            dropout=0.3, bidirectional=True)
        self.attn = nn.Linear(512, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(512), nn.Dropout(0.3), nn.Linear(512, 128),
            nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 1)
        )
    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        w = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1).unsqueeze(-1)
        context = (lstm_out * w).sum(dim=1)
        return self.classifier(context).squeeze(-1)

# ── Load artifacts once at startup ───────────────────────────────────────────
print("Loading model artifacts...")
with open(MODEL_DIR / 'feature_columns.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

n_cols = len(feature_cols)
n_timesteps = max(1, math.ceil(n_cols / TARGET_NF))
n_features = TARGET_NF
pad_to = n_timesteps * n_features

model = EmberLSTM(n_features, n_timesteps).to(DEVICE)
model.load_state_dict(torch.load(MODEL_DIR / 'ember_lstm_best.pt', map_location=DEVICE))
model.eval()

print(f"Ready: {n_cols} features → {n_timesteps}×{n_features} | Device: {DEVICE}")

# ── Request/Response schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: list[float]

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    x = np.array(req.features, dtype=np.float32)

    # Trim/pad to exact column count expected by scaler
    if len(x) < n_cols:
        x = np.pad(x, (0, n_cols - len(x)), constant_values=0.0)
    elif len(x) > n_cols:
        x = x[:n_cols]

    # Scale
    x_scaled = scaler.transform(x.reshape(1, -1))[0]

    # Pad to multiple of TARGET_NF (e.g. 689 → 704)
    if len(x_scaled) < pad_to:
        x_scaled = np.pad(x_scaled, (0, pad_to - len(x_scaled)), constant_values=0.0)

    # Reshape for LSTM: (1, timesteps, features)
    tensor = torch.tensor(x_scaled.reshape(n_timesteps, n_features), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == 'cuda')):
            prob = torch.sigmoid(model(tensor)).item()

    # Confidence labels expected by the extension
    if prob > 0.9:
        conf = 'high'
    elif prob > 0.7:
        conf = 'medium'
    elif prob > 0.5:
        conf = 'low'
    else:
        conf = 'benign'

    return {
        "is_malicious": prob > 0.5,
        "confidence": conf,
        "malicious_probability": prob
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)