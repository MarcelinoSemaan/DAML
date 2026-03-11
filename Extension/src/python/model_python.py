# src/python/model_server.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import json
import asyncio
from pathlib import Path
import sys
import logging

# Add your model import here
# from your_model import MalwareDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DAML Model API", version="2.1.4")

# CORS for VS Code webview communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ScanRequest(BaseModel):
    path: str
    scan_type: str = "full"  # "full", "quick", "file"

class ThreatInfo(BaseModel):
    file: str
    threat_type: str
    confidence: float
    severity: str  # "low", "medium", "high", "critical"
    timestamp: str

class ScanResult(BaseModel):
    scan_id: str
    total_files: int
    threats_found: int
    safe_files: int
    score: float
    threats: List[ThreatInfo]
    summary: str

class HealthStatus(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    engine_version: str = "2.1.4"

# Global model instance (loaded once)
model = None

class MalwareDetector:
    """Placeholder for your actual model - replace with your implementation"""
    
    def __init__(self):
        self.loaded = False
        self.version = "1.0.0"
        
    def load(self):
        """Load your model weights here"""
        logger.info("Loading model...")
        # Load your actual model
        # self.model = torch.load('weights.pth')
        self.loaded = True
        logger.info("Model loaded successfully")
        
    def predict(self, file_path: str) -> Dict[str, Any]:
        """Run inference on a file"""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        # Replace with actual inference
        # result = self.model.predict(file_path)
        
        # Simulated response for testing
        import random
        from datetime import datetime
        
        is_threat = random.random() > 0.85  # 15% threat rate for demo
        
        return {
            "is_threat": is_threat,
            "confidence": random.uniform(0.7, 0.99) if is_threat else random.uniform(0.01, 0.3),
            "family": random.choice(["trojan", "ransomware", "spyware", "adware"]) if is_threat else None,
            "severity": random.choice(["medium", "high", "critical"]) if is_threat else "safe",
            "features": {
                "entropy": random.uniform(3.0, 8.0),
                "pe_imports": random.randint(10, 500),
                "strings_count": random.randint(100, 10000)
            }
        }

def get_model():
    global model
    if model is None:
        model = MalwareDetector()
        model.load()
    return model

@app.on_event("startup")
async def startup():
    get_model()

@app.get("/health", response_model=HealthStatus)
async def health():
    m = get_model()
    return HealthStatus(
        status="running",
        model_loaded=m.loaded,
        model_version=m.version,
        engine_version="2.1.4"
    )

@app.post("/scan", response_model=ScanResult)
async def scan(request: ScanRequest, background_tasks: BackgroundTasks):
    """Scan a file or directory"""
    logger.info(f"Scan requested: {request.path}, type: {request.scan_type}")
    
    m = get_model()
    path = Path(request.path)
    
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {request.path}")
    
    # Collect files to scan
    files_to_scan = []
    if path.is_file():
        files_to_scan = [path]
    elif path.is_dir():
        # Recursive scan with filtering
        extensions = {'.exe', '.dll', '.bin', '.dat', '.jsonl'} if request.scan_type == "full" else {'.exe'}
        for ext in extensions:
            files_to_scan.extend(path.rglob(f"*{ext}"))
        files_to_scan = files_to_scan[:100]  # Limit for demo
    
    # Run scans
    threats = []
    safe_count = 0
    
    for file_path in files_to_scan:
        try:
            result = m.predict(str(file_path))
            
            if result["is_threat"]:
                from datetime import datetime
                threats.append(ThreatInfo(
                    file=str(file_path),
                    threat_type=result["family"] or "unknown",
                    confidence=result["confidence"],
                    severity=result["severity"],
                    timestamp=datetime.now().isoformat()
                ))
            else:
                safe_count += 1
                
        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")
    
    # Calculate score (0-100, higher is safer)
    total = len(files_to_scan)
    threat_count = len(threats)
    score = max(0, 100 - (threat_count / max(total, 1) * 100) * 2)
    
    scan_id = f"scan_{asyncio.get_event_loop().time()}"
    
    return ScanResult(
        scan_id=scan_id,
        total_files=total,
        threats_found=threat_count,
        safe_files=safe_count,
        score=round(score, 1),
        threats=threats,
        summary=f"Scanned {total} files, found {threat_count} threats" if threat_count > 0 else f"All {total} files clean"
    )

@app.post("/scan/file")
async def scan_single(file_path: str):
    """Quick scan of a single file"""
    m = get_model()
    result = m.predict(file_path)
    return {
        "file": file_path,
        **result
    }

@app.get("/stats")
async def get_stats():
    """Get historical statistics for dashboard charts"""
    # In production, read from database or log files
    return {
        "daily": {
            "labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "safe": [150, 200, 180, 220, 170, 90, 100],
            "threats": [2, 5, 1, 0, 4, 1, 1]
        },
        "top_families": [
            {"name": "trojan", "count": 45},
            {"name": "ransomware", "count": 23},
            {"name": "spyware", "count": 12}
        ]
    }

@app.post("/resolve")
async def resolve_threats(threat_ids: List[str]):
    """Mark threats as resolved"""
    return {"resolved": len(threat_ids), "status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
