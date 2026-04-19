import torch
import torch.nn as nn
import os

# =============================================================================
# Your Original Architecture (must match checkpoint exactly)
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

    def forward(self, x, return_metadata=False):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1).unsqueeze(-1)
        context = (lstm_out * weights).sum(dim=1)
        logit = self.classifier(context).squeeze(-1)
        
        if return_metadata:
            prob = torch.sigmoid(logit)
            confidence = torch.where(prob > 0.5, prob, 1 - prob)
            return logit, confidence, "clean", x.mean().item()
        return logit

# =============================================================================
# EMBER-Specific Hardened Wrapper
# =============================================================================
class EMBERGuardedLSTM(nn.Module):
    """
    Wraps EmberLSTM with guards specific to EMBER2024 feature semantics:
      - Rejects all-zero inputs (impossible for real PE files)
      - Rejects constant-sequence inputs (no temporal variance in feature windows)
      - Rejects extreme values outside EMBER's natural range
      - Applies temperature scaling to reduce sparsity-exploitation confidence
    """
    def __init__(self, base_model, temperature=2.0, 
                 max_valid_feature=50.0, min_variance=0.05):
        super().__init__()
        self.base_model = base_model
        self.temperature = nn.Parameter(torch.tensor([temperature]))
        self.max_valid_feature = max_valid_feature
        self.min_variance = min_variance
        
    def _ember_ood_check(self, x):
        """
        EMBER2024-specific validity checks.
        Real PE files never produce all-zero feature vectors.
        """
        # Check 1: All zeros (physically impossible for PE files)
        if torch.allclose(x, torch.zeros_like(x), atol=1e-6):
            return True, "all_zeros", 0.0
            
        # Check 2: No temporal variance across the 24 timesteps
        # In EMBER, different feature groups (histograms, sections, strings)
        # should vary across the reshaped windows
        t_std = x.std(dim=1).mean()
        if t_std < self.min_variance:
            return True, "flat_sequence", 0.0
            
        # Check 3: Values outside EMBER's natural range
        # EMBER features are mostly counts, entropies [0,1], and histograms [0,1]
        if x.abs().max() > self.max_valid_feature:
            return True, "out_of_range", 0.0
            
        # Check 4: Sequence mean is pathologically low
        # Real PE files have non-trivial byte entropy and string counts
        seq_mean = x.mean()
        if seq_mean < 0.01:
            return True, "near_zero_mean", seq_mean.item()
            
        return False, "clean", seq_mean.item()

    def forward(self, x, return_metadata=False):
        is_ood, reason, seq_mean = self._ember_ood_check(x)
        
        if is_ood:
            batch_size = x.size(0)
            neutral = torch.zeros(batch_size, device=x.device)
            if return_metadata:
                return neutral, torch.zeros(batch_size), reason, seq_mean
            return neutral
        
        logits = self.base_model(x)
        # Temperature scaling: T > 1 reduces the sparsity shortcut's confidence
        scaled_logits = logits / self.temperature.abs()
        probs = torch.sigmoid(scaled_logits)
        confidence = torch.where(probs > 0.5, probs, 1 - probs)
        
        if return_metadata:
            return scaled_logits, confidence, "clean", seq_mean
        return scaled_logits

# =============================================================================
# Configuration
# =============================================================================
CHECKPOINT = "C:/Users/Dell/Documents/GitHub/DAML/AI Model/ember_lstm_best.pt"
EXPORT_DIR = "C:/Users/Dell/Documents/GitHub/DAML/AI Model"
N_FEATURES = 64
N_TIMESTEPS = 24

os.makedirs(EXPORT_DIR, exist_ok=True)

# =============================================================================
# Load & Wrap
# =============================================================================
base_model = EmberLSTM(N_FEATURES, N_TIMESTEPS)
base_model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
base_model.eval()

# Wrap: Temperature=2.0 pushes overconfident sparsity predictions toward 0.5
model = EMBERGuardedLSTM(base_model, temperature=2.0)
model.eval()

# =============================================================================
# Diagnostic Suite: Original vs Guarded
# =============================================================================
def run_diagnostics(model, label="Model"):
    tests = {
        "All Zeros (Impossible PE)": torch.zeros(1, N_TIMESTEPS, N_FEATURES),
        "All Ones (Dense)": torch.ones(1, N_TIMESTEPS, N_FEATURES),
        "Random Normal": torch.randn(1, N_TIMESTEPS, N_FEATURES),
        "Random Uniform [0,1]": torch.rand(1, N_TIMESTEPS, N_FEATURES),
        "Large Values (100)": torch.full((1, N_TIMESTEPS, N_FEATURES), 100.0),
        "Small Values (-100)": torch.full((1, N_TIMESTEPS, N_FEATURES), -100.0),
        "Alternating Pattern": torch.tensor([[[1.0 if (i+j) % 2 == 0 else -1.0 
                                              for j in range(N_FEATURES)] 
                                             for i in range(N_TIMESTEPS)]]),
        "Realistic EMBER (mu=0.3, std=0.2)": torch.randn(1, N_TIMESTEPS, N_FEATURES) * 0.2 + 0.3,
        "Sparse EMBER (mu=0.02, std=0.05)": torch.randn(1, N_TIMESTEPS, N_FEATURES).abs() * 0.05,
    }
    
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"{'Input':<35s} {'Logit':>8s} {'Prob':>8s} {'Conf':>8s} {'Status'}")
    print("-" * 70)
    
    with torch.no_grad():
        for name, x in tests.items():
            out = model(x, return_metadata=True)
            logit, conf, reason, seq_mean = out
            prob = torch.sigmoid(logit).item()
            
            if reason != "clean":
                status = f"[REFUSED: {reason}]"
            elif 0.4 <= prob <= 0.6:
                status = "[UNCERTAIN]"
            elif prob > 0.85:
                status = "[HIGH CONF MALWARE]"
            elif prob < 0.15:
                status = "[HIGH CONF BENIGN]"
            else:
                status = "[MODERATE]"
            
            print(f"{name:<35s} {logit.item():>8.4f} {prob:>8.4f} {conf.item():>8.4f} {status}")

run_diagnostics(base_model, "BASE MODEL (Sparsity Shortcut Exploited)")
run_diagnostics(model, "GUARDED MODEL (Temperature + OOD Rejection)")

# =============================================================================
# Export 1: State Dict (resume training / fine-tune on EMBER2024 challenge set)
# =============================================================================
weights_path = os.path.join(EXPORT_DIR, "ember_lstm_weights.pt")
torch.save(model.state_dict(), weights_path)
print(f"\n[EXPORTED] {weights_path}")
print("           Use: model.load_state_dict(torch.load(path))")

# =============================================================================
# Export 2: Full Model (Python inference, architecture embedded)
# =============================================================================
full_path = os.path.join(EXPORT_DIR, "ember_lstm_full.pt")
torch.save(model, full_path)
print(f"[EXPORTED] {full_path}")
print("           Use: model = torch.load(path)")

# =============================================================================
# Export 3: TorchScript (C++ / deployment / no Python dependency)
# =============================================================================
scripted_path = os.path.join(EXPORT_DIR, "ember_lstm_scripted.pt")
try:
    # Scripting preserves the if/else OOD logic; tracing would not
    scripted_model = torch.jit.script(model)
    scripted_model.save(scripted_path)
    print(f"[EXPORTED] {scripted_path}")
    print("           Use: model = torch.jit.load(path)")
    
    # Verify numerical parity on valid inputs only
    test_x = torch.randn(5, N_TIMESTEPS, N_FEATURES).abs() * 0.3 + 0.1
    with torch.no_grad():
        y_orig = model(test_x)
        y_script = scripted_model(test_x)
    diff = torch.abs(y_orig - y_script).max().item()
    print(f"[VERIFY]   TorchScript parity max diff: {diff:.2e}")
except Exception as e:
    print(f"[ERROR]    TorchScript export failed: {e}")