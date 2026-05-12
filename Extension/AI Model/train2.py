import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Settings from your daml.py
DATA_DIR = '/home/marcelino/Desktop/Data/Dataset/Combined-DB'
MEMMAP_DIR = Path(DATA_DIR) / 'memmap_cache'

def generate_scaler_only():
    # 1. Load metadata to get the shape
    with open(MEMMAP_DIR / 'train_meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    
    shape = meta['shape']
    X_path = MEMMAP_DIR / 'train_X.dat'
    
    print(f"Fitting scaler for {shape[0]} samples...")
    
    # 2. Re-run the incremental fit logic
    scaler = StandardScaler()
    X = np.memmap(X_path, dtype='float32', mode='r', shape=shape)
    
    indices = np.arange(shape[0])
    np.random.shuffle(indices)
    
    chunksize = 5000
    for start in tqdm(range(0, shape[0], chunksize), desc="Partial fit"):
        end = min(start + chunksize, shape[0])
        chunk = X[indices[start:end]]
        scaler.partial_fit(chunk)
    
    # 3. Save the new pkl
    with open(MEMMAP_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nSuccess! Scaler saved to {MEMMAP_DIR}/scaler.pkl")

if __name__ == "__main__":
    generate_scaler_only()