import torch
import torch.nn as nn
import os

# =============================================================================
# Model Architecture (must match training exactly)
# =============================================================================
class EmberLSTM(nn.Module):
    def __init__(self, n_features, n_timesteps):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 128),
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

# =============================================================================
# Configuration
# =============================================================================
CHECKPOINT = "AI Model/ember_lstm_best.pt"
EXPORT_DIR = "AI Model"
N_FEATURES = 64
N_TIMESTEPS = 24

os.makedirs(EXPORT_DIR, exist_ok=True)

# =============================================================================
# Load Model
# =============================================================================
model = EmberLSTM(N_FEATURES, N_TIMESTEPS)
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
model.eval()

print("=" * 70)
print("ORIGINAL MODEL DIAGNOSTICS")
print("=" * 70)

# =============================================================================
# Diagnostic Test Suite
# =============================================================================
def run_diagnostics(model, label="Model"):
    """Run controlled inputs through model and report probabilities."""
    tests = {
        "All Zeros": torch.zeros(1, N_TIMESTEPS, N_FEATURES),
        "All Ones": torch.ones(1, N_TIMESTEPS, N_FEATURES),
        "Random Normal": torch.randn(1, N_TIMESTEPS, N_FEATURES),
        "Random Uniform [0,1]": torch.rand(1, N_TIMESTEPS, N_FEATURES),
        "Large Values (100)": torch.full((1, N_TIMESTEPS, N_FEATURES), 100.0),
        "Small Values (-100)": torch.full((1, N_TIMESTEPS, N_FEATURES), -100.0),
        "Alternating Pattern": torch.tensor([[[1.0 if (i+j) % 2 == 0 else -1.0 
                                              for j in range(N_FEATURES)] 
                                             for i in range(N_TIMESTEPS)]]),
    }
    
    print(f"\n--- {label} ---")
    with torch.no_grad():
        for name, x in tests.items():
            raw_out = model(x)
            prob = torch.sigmoid(raw_out).item()
            status = ""
            if name == "All Zeros":
                if 0.45 <= prob <= 0.55:
                    status = "  [OK: well-calibrated bias]"
                elif prob > 0.9:
                    status = "  [CRITICAL: broken/overfitted]"
                elif prob < 0.1:
                    status = "  [CRITICAL: broken/inverted]"
            print(f"{name:25s}: {prob:.6f}{status}")
    
    # Check for NaN/Inf
    print(f"\nNumerical Stability Check:")
    x_test = torch.randn(10, N_TIMESTEPS, N_FEATURES)
    with torch.no_grad():
        out = model(x_test)
    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    print(f"  NaN in output: {has_nan}")
    print(f"  Inf in output: {has_inf}")

run_diagnostics(model, "Original Model")


# =============================================================================
# Interpretation Guide
# =============================================================================
print("\n" + "=" * 70)
print("INTERPRETATION GUIDE")
print("=" * 70)
print("""
All Zeros ~0.5     : Model has healthy bias. If real predictions are wrong,
                     your feature scaling/extraction is likely the issue.
                     
All Zeros >0.9     : Model is broken or severely overfitted. It predicts 
                     positive regardless of input. Retrain with regularization.
                     
All Zeros <0.1     : Model is broken or training labels are inverted.
                     Check loss function and label encoding.

Large Values       : Should not crash (NaN/Inf). If it outputs 0.0 or 1.0 
                     exactly, that's expected due to sigmoid saturation.

Random inputs      : Should vary between runs for 'Random Normal/Uniform'.
                     If always the same, random seed is fixed somewhere.
""")