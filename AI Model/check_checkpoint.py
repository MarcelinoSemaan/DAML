import torch

# Update this path to your Actual checkpoint location
CHECKPOINT_PATH = "C:/Users/Dell/Documents/GitHub/DAML/AI Model/ember_lstm_best.pt"

print(f"Loading checkpoint from: {CHECKPOINT_PATH}")

try:
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    print("\nCheckpoint layers found:")
    for name, param in state_dict.items():
        if "input_proj.0.weight" in name:
            in_features = param.shape[1]
            out_features = param.shape[0]
            print(f"  First layer: Linear(in={in_features}, out={out_features})")
            print(f"  Your model was trained with: {in_features} input features")
            break
    
except Exception as e:
    print(f"Error loading checkpoint: {e}")