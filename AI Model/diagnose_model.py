import torch
import torch.nn as nn
import numpy as np

# Your model architecture
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

# Load your checkpoint
CHECKPOINT = "C:/Users/Dell/Documents/GitHub/DAML/AI Model/ember_lstm_best.pt"
model = EmberLSTM(64, 24)
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
model.eval()

print("=== Diagnostic: What does the model output for controlled inputs? ===\n")

tests = {
    "All Zeros": torch.zeros(1, 24, 64),
    "All Ones": torch.ones(1, 24, 64),
    "Random Normal": torch.randn(1, 24, 64),
    "Random Uniform [0,1]": torch.rand(1, 24, 64),
    "Large Values (100)": torch.full((1, 24, 64), 100.0),
}

with torch.no_grad():
    for name, x in tests.items():
        prob = torch.sigmoid(model(x)).item()
        print(f"{name:25s}: {prob:.6f}")

print("\n=== Interpretation ===")
print("If 'All Zeros' is ~0.5, the model is fine but your features are wrong.")
print("If 'All Zeros' is >0.9, the model is broken/overfitted and useless.")