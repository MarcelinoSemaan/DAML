from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from model import EmberLSTM, N_TIMESTEPS, N_FEATURES

app = FastAPI(title="EmberLSTM Malware Detection API")

# Load model once at startup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmberLSTM().to(DEVICE)

# Load your trained weights
try:
    model.load_state_dict(torch.load('ember_lstm_best.pt', map_location=DEVICE))
    model.eval()
    print(f"✅ Model loaded on {DEVICE}")
except FileNotFoundError:
    print("⚠️ No trained weights found - model is untrained")

class PredictionRequest(BaseModel):
    features: list  # Flat list of 2568 floats, or nested 24×107

class PredictionResponse(BaseModel):
    malicious_probability: float
    is_malicious: bool
    confidence: str  # "low", "medium", "high"

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Convert to tensor and reshape
        features = np.array(request.features, dtype=np.float32)
        
        # Handle both flat (2568,) and pre-shaped (24, 107) inputs
        if features.size == N_TIMESTEPS * N_FEATURES:
            features = features.reshape(N_TIMESTEPS, N_FEATURES)
        elif features.shape != (N_TIMESTEPS, N_FEATURES):
            raise ValueError(f"Expected {N_TIMESTEPS * N_FEATURES} features, got {features.size}")
        
        # Clean NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Inference
        x = torch.tensor(features).unsqueeze(0).to(DEVICE)  # Add batch dim
        
        with torch.no_grad():
            with torch.amp.autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
                logits = model(x)
                prob = torch.sigmoid(logits).item()
        
        # Response formatting
        confidence = "high" if prob > 0.8 or prob < 0.2 else "medium" if prob > 0.6 or prob < 0.4 else "low"
        
        return PredictionResponse(
            malicious_probability=round(prob, 4),
            is_malicious=prob > 0.5,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE), "model_loaded": model is not None}

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000