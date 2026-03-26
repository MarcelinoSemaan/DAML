import numpy as np, torch, thrember
from pathlib import Path
from train import EmberLSTM, N_TIMESTEPS, N_FEATURES, DEVICE

DATA_DIR = Path('/home/marcelino/Desktop/Data/Dataset/Vectorised-Data')
X = np.memmap(DATA_DIR / "X_train.dat", dtype=np.float32, mode='r').reshape(-1, thrember.PEFeatureExtractor().dim)
y = np.memmap(DATA_DIR / "y_train.dat", dtype=np.int32, mode='r')

batch_X = np.stack([X[i].reshape(N_TIMESTEPS, N_FEATURES) for i in range(8)])
batch_y = np.array([(int(y[i] > 0)) for i in range(8)], dtype=np.float32)

batch_X = torch.tensor(batch_X, dtype=torch.float32).to(DEVICE)
batch_y = torch.tensor(batch_y, dtype=torch.float32).to(DEVICE)

model = EmberLSTM().to(DEVICE)
model.eval()

with torch.no_grad():
    logits = model(batch_X)
    print("logits shape:", logits.shape)
    print("logits finite:", torch.isfinite(logits).all().item())
    print("logits min/max/mean/std:", logits.min().item(), logits.max().item(), logits.mean().item(), logits.std().item())
    loss = torch.nn.BCEWithLogitsLoss()(logits, batch_y)
    print("loss:", loss.item(), "loss finite:", torch.isfinite(loss).item())