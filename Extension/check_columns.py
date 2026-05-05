import pickle
from pathlib import Path

# Adjust this path if your feature_columns.pkl is in a different location
FEATURE_COLS_PATH = Path(r"C:\Users\Dell\Downloads\DAML-LSTM-Scaler-version\DAML-LSTM-Scaler-version\AI Model\feature_columns.pkl")

with open(FEATURE_COLS_PATH, "rb") as f:
    cols = pickle.load(f)

# Check first few import-related columns
import_cols = [c for c in cols if c.startswith("imports.")]
print(f"Total import columns: {len(import_cols)}")
print("First 10 import columns:", import_cols[:10])
print("\nLast 10 import columns:", import_cols[-10:])

# Also check a few non-import columns to verify overall structure
print("\nFirst 10 columns overall:", cols[:10])
print(f"\nTotal columns: {len(cols)}")