import pickle
import numpy as np

model_dir = r"C:\Users\Dell\Downloads\DAML-LSTM-Scaler-version\DAML-LSTM-Scaler-version\AI Model"

with open(f"{model_dir}\\feature_columns.pkl", "rb") as f:
    cols = pickle.load(f)

with open(f"{model_dir}\\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# These cols cannot be extracted from PE — fill with training mean
VT_ONLY_COLS = {
    "first_submission_date",
    "last_analysis_date",
    "week_id",
    "authenticode.latest_signing_time",
    "authenticode.signing_time_diff",
}

vt_means = {}
for i, col in enumerate(cols):
    if col in VT_ONLY_COLS:
        vt_means[col] = float(scaler.mean_[i])
        print(f"{col}: mean={scaler.mean_[i]:.2f}")

with open(f"{model_dir}\\vt_means.pkl", "wb") as f:
    pickle.dump(vt_means, f)

print("\nSaved vt_means.pkl")