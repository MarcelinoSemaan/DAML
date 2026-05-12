# DAML Protection

AI-powered malware detection for VS Code using EMBER-LSTM.

## Requirements
- Python 3.8+
- FastAPI server running on `localhost:8000`
- Install server deps: `pip install fastapi uvicorn torch numpy thrember`

## Usage
1. Start the DAML API server: `uvicorn main:app --host 0.0.0.0 --port 8000`
2. Open the DAML Protection sidebar in VS Code
3. Click "Scan Workspace" or "Scan PE File" to analyze binaries