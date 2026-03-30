import os
# Set the backend BEFORE importing keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="DAML Model Server")

# 1. Enable CORS (Cross-Origin Resource Sharing)
# This prevents "Block by CORS policy" errors in your extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Safety check for the model file
MODEL_PATH = "my_lstm_model.keras"
if not os.path.exists(MODEL_PATH):
    print(f"CRITICAL ERROR: {MODEL_PATH} not found in {os.getcwd()}")
    model = None
else:
    print(f"Loading {MODEL_PATH} using {os.environ['KERAS_BACKEND']}...")
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None

class ThreatData(BaseModel):
    raw_sequence: list[float]

@app.post("/analyze")
async def analyze_threat(data: ThreatData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
        
    try:
        # Expected size: 10 timesteps * 2568 features = 25680 elements
        input_array = np.array(data.raw_sequence, dtype=np.float32)
        
        # Verify length before reshaping to avoid a server crash
        if input_array.size != 25680:
             raise ValueError(f"Expected 25680 values, got {input_array.size}")

        reshaped_input = input_array.reshape(1, 10, 2568)
        
        # verbose=0 keeps the terminal clean
        prediction = model.predict(reshaped_input, verbose=0)
        
        return {
            "status": "success", 
            "threat_score": prediction.tolist()
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)