import torch
import torch.nn as nn

class EmberLSTM(nn.Module):
    def __init__(self, n_features, n_timesteps):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.attn = nn.Linear(512, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(512), nn.Dropout(0.3), nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 1)
        )
    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1).unsqueeze(-1)
        context = (lstm_out * weights).sum(dim=1)
        return self.classifier(context).squeeze(-1)

print("=== ember_lstm_full.pt (forcing load) ===")
try:
    # PyTorch 2.6+ requires weights_only=False for full models
    model = torch.load("ember_lstm_full.pt", map_location="cpu", weights_only=False)
    model.eval()
    n_features = model.input_proj[0].in_features
    print(f"  Features: {n_features}")
    
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.zeros(1, 24, n_features))).item()
    print(f"  Zeros score: {prob:.6f}")
    
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.randn(1, 24, n_features))).item()
    print(f"  Random score: {prob:.6f}")
    
    status = "✅ GOOD" if 0.3 < prob < 0.7 else "❌ BROKEN"
    print(f"  Status: {status}")
    
except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== ember_lstm_scripted.pt ===")
try:
    model = torch.jit.load("ember_lstm_scripted.pt", map_location="cpu")
    model.eval()
    
    # Scripted models don't expose architecture easily, test both 64 and 107
    for feat in [64, 107]:
        try:
            with torch.no_grad():
                prob = torch.sigmoid(model(torch.zeros(1, 24, feat))).item()
            print(f"  Features {feat}: Zeros={prob:.6f}")
            status = "✅ GOOD" if 0.3 < prob < 0.7 else "❌ BROKEN"
            print(f"    Status: {status}")
            break
        except RuntimeError:
            print(f"  Features {feat}: Wrong input size")
            
except Exception as e:
    print(f"  ERROR: {e}")