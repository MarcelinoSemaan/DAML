import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import thrember

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = '/home/marcelino/Desktop/Data/Dataset/Vectorised-Data'
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 1024
EPOCHS      = 15
N_WORKERS   = 4
N_TIMESTEPS = 24    # 24 × 107 = 2568 exactly
N_FEATURES  = 107

print(f"Device : {DEVICE}")
print(f"Reshape: {N_TIMESTEPS} × {N_FEATURES} = {N_TIMESTEPS * N_FEATURES}")
assert N_TIMESTEPS * N_FEATURES == 2568, "Reshape mismatch!"

if DEVICE.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(data_dir):
    data_dir  = Path(data_dir)
    extractor = thrember.PEFeatureExtractor()
    ndim      = extractor.dim
    X_tr = np.memmap(data_dir / "X_train.dat", dtype=np.float32, mode="r").reshape(-1, ndim)
    y_tr = np.memmap(data_dir / "y_train.dat", dtype=np.int32,   mode="r")
    X_te = np.memmap(data_dir / "X_test.dat",  dtype=np.float32, mode="r").reshape(-1, ndim)
    y_te = np.memmap(data_dir / "y_test.dat",  dtype=np.int32,   mode="r")
    return X_tr, y_tr, X_te, y_te

x_train, y_train, x_test, y_test = load_data(DATA_DIR)
print(f"Train  : {x_train.shape}  Test: {x_test.shape}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class EmberDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy().reshape(N_TIMESTEPS, N_FEATURES)
        # Replace any NaN/Inf in raw features before they enter the model
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = int(self.y[idx] > 0)
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))

train_loader = DataLoader(EmberDataset(x_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=N_WORKERS, pin_memory=True)
test_loader  = DataLoader(EmberDataset(x_test, y_test),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=N_WORKERS, pin_memory=True)

print(f"Train batches: {len(train_loader)}  Test batches: {len(test_loader)}")

import thrember, numpy as np, os
DATA_DIR = '/home/marcelino/Desktop/Data/Dataset/Vectorised-Data'
e = thrember.PEFeatureExtractor()
ndim = e.dim
path = os.path.join(DATA_DIR, "X_train.dat")
X = np.memmap(path, dtype=np.float32, mode='r')
print("extractor.dim:", ndim)
print("X_train.dat size (elements):", X.size)
print("rows (size // dim):", X.size // ndim)
print("remainder (size % dim):", X.size % ndim)
import numpy as np, os
DATA_DIR = '/home/marcelino/Desktop/Data/Dataset/Vectorised-Data'
path = os.path.join(DATA_DIR, "X_train.dat")
nd = 2568
X = np.memmap(path, dtype=np.float32, mode='r')[:nd*5].reshape(5, nd)
print("finite_all:", np.isfinite(X).all())
print("nan_count:", int(np.isnan(X).sum()))
print("inf_count:", int(np.isinf(X).sum()))
print("rows checked:", X.shape[0], "cols:", X.shape[1])
import numpy as np, os
DATA_DIR = '/home/marcelino/Desktop/Data/Dataset/Vectorised-Data'
path = os.path.join(DATA_DIR, "X_train.dat")
nd = 2568
X = np.memmap(path, dtype=np.float32, mode='r')[:nd*5].reshape(5, nd)
print("finite_all:", np.isfinite(X).all())
print("nan_count:", int(np.isnan(X).sum()))
print("inf_count:", int(np.isinf(X).sum()))
print("rows checked:", X.shape[0], "cols:", X.shape[1])
import numpy as np, torch
from pathlib import Path
import thrember

DATA_DIR = Path('/home/marcelino/Desktop/Data/Dataset/Vectorised-Data')
# memmap labels
y = np.memmap(DATA_DIR / "y_train.dat", dtype=np.int32, mode='r')
print("y.size:", y.size, "unique sample:", np.unique(y)[:10])

# instantiate extractor and dataset
from train import EmberDataset, N_TIMESTEPS, N_FEATURES  # adjust import if filename differs
extractor = thrember.PEFeatureExtractor()
print("extractor.dim:", extractor.dim)

# Inspect first 5 items from EmberDataset
X = np.memmap(DATA_DIR / "X_train.dat", dtype=np.float32, mode='r').reshape(-1, extractor.dim)
ds = EmberDataset(X, y)
for i in range(5):
    x_tensor, y_tensor = ds[i]
    x = x_tensor.numpy()
    print(f"idx={i} y={y_tensor.item()} shape={x.shape} finite_all={np.isfinite(x).all()} nan={int(np.isnan(x).sum())} inf={int(np.isinf(x).sum())}")
import numpy as np, torch
from pathlib import Path
from train import EmberDataset, EmberLSTM, N_TIMESTEPS, N_FEATURES, DEVICE, BATCH_SIZE
import thrember

DATA_DIR = Path('/home/marcelino/Desktop/Data/Dataset/Vectorised-Data')
X = np.memmap(DATA_DIR / "X_train.dat", dtype=np.float32, mode='r')
y = np.memmap(DATA_DIR / "y_train.dat", dtype=np.int32, mode='r')
X = X.reshape(-1, thrember.PEFeatureExtractor().dim)

ds = EmberDataset(X, y)
xb, yb = ds[0:BATCH_SIZE] if hasattr(ds, '__getitem__') and isinstance(ds[0], tuple) else None

# If EmberDataset returns tensors per item, build a batch manually:
if xb is None:
    batch_X = np.stack([ds[i][0].numpy() for i in range(min(8, len(ds)))])
    batch_y = np.array([ds[i][1].item() for i in range(min(8, len(ds)))]).astype(np.float32)
else:
    batch_X, batch_y = xb, yb

batch_X = torch.tensor(batch_X, dtype=torch.float32).to(DEVICE)
batch_y = torch.tensor(batch_y, dtype=torch.float32).to(DEVICE)

model = EmberLSTM().to(DEVICE)
model.eval()

with torch.no_grad():
    logits = model(batch_X)
    print("logits shape:", logits.shape)
    print("logits finite:", torch.isfinite(logits).all().item())
    print("logits stats:", logits.min().item(), logits.max().item(), logits.mean().item(), logits.std().item())
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(logits, batch_y)
    print("loss:", loss.item(), "isfinite:", torch.isfinite(loss).item())



# ── Model ─────────────────────────────────────────────────────────────────────
class EmberLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(N_FEATURES, 128),
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
        x           = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        weights     = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1).unsqueeze(-1)
        context     = (lstm_out * weights).sum(dim=1)
        return self.classifier(context).squeeze(-1)

model    = EmberLSTM().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {n_params:,}")

# ── Training loop ─────────────────────────────────────────────────────────────
nan_batches = 0  # global counter so we can report at end of epoch

def run_epoch(model, loader, criterion, optimizer, scaler, train=True, desc=""):
    nan_batches = 0
    model.train() if train else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []

    bar = tqdm(loader, desc=desc, leave=False, bar_format="{l_bar}{bar:30}{r_bar}")

    for xb, yb in bar:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with torch.amp.autocast('cuda'):
            logits = model(xb)
            logits = torch.clamp(logits, -1e6, 1e6)
            loss = criterion(logits, yb)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            if train:
                optimizer.zero_grad(set_to_none=True)
            continue

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        
                # compute logits & loss
        with torch.amp.autocast('cuda'):
            logits = model(xb)
            logits = torch.clamp(logits, -1e6, 1e6)
            loss = criterion(logits, yb)

        # if loss is NaN/Inf, print diagnostics then skip
        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            # diagnostics
            try:
                lmin = logits.min().detach().cpu().item()
                lmax = logits.max().detach().cpu().item()
                lmean = logits.mean().detach().cpu().item()
                finite_logits = torch.isfinite(logits).all().item()
            except Exception:
                lmin = lmax = lmean = None
                finite_logits = False
            print(f"Skipping batch #{nan_batches}: loss={loss.item() if isinstance(loss, torch.Tensor) else loss} | logits_finite={finite_logits} | logits_min={lmin} max={lmax} mean={lmean} | labels_finite={np.isfinite(yb.cpu().numpy()).all()} | labels_unique={np.unique(yb.cpu().numpy())[:10]}")
            if train:
                optimizer.zero_grad(set_to_none=True)
            continue

        total_loss += loss.item()
        safe_probs = torch.sigmoid(torch.clamp(logits, -20, 20)).detach().cpu().numpy()
        all_probs.extend(safe_probs)
        all_labels.extend(yb.cpu().numpy())

    if len(all_labels) == 0:
        print("⚠️  Warning: No valid batches processed in this epoch!")
        return float('nan'), 0.0, 0.0, 0.0

    labels = np.array(all_labels)
    probs  = np.nan_to_num(np.array(all_probs), nan=0.5, posinf=1.0, neginf=0.0)
    preds  = (probs > 0.5).astype(int)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    return (total_loss / max(len(loader) - nan_batches, 1),
            accuracy_score(labels, preds),
            auc,
            f1_score(labels, preds, zero_division=0))
            
# ── Optimiser / scheduler / scaler ───────────────────────────────────────────
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = OneCycleLR(optimizer, max_lr=1e-4,
                       steps_per_epoch=len(train_loader), epochs=EPOCHS,
                       pct_start=0.3, div_factor=10, final_div_factor=100)
scaler = torch.amp.GradScaler()
history   = []
best_auc  = 0.0

# ── Train ─────────────────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    print(f"\n── Epoch {epoch}/{EPOCHS} ──")

    tr = run_epoch(model, train_loader, criterion, optimizer, scaler,
                   train=True,  desc=f"Epoch {epoch}/{EPOCHS} Train")
    scheduler.step()

    with torch.no_grad():
        va = run_epoch(model, test_loader, criterion, optimizer, scaler,
                       train=False, desc=f"Epoch {epoch}/{EPOCHS} Val  ")

    print(f"  Train  loss={tr[0]:.4f}  acc={tr[1]:.4f}  auc={tr[2]:.4f}  f1={tr[3]:.4f}")
    print(f"  Val    loss={va[0]:.4f}  acc={va[1]:.4f}  auc={va[2]:.4f}  f1={va[3]:.4f}")

    history.append(dict(tr_loss=tr[0], tr_acc=tr[1], tr_auc=tr[2], tr_f1=tr[3],
                        va_loss=va[0], va_acc=va[1], va_auc=va[2], va_f1=va[3]))

    if va[2] > best_auc:
        best_auc = va[2]
        torch.save(model.state_dict(), 'ember_lstm_best.pt')
        print(f"  ✓ Saved (AUC={best_auc:.4f})")

print(f"\nDone. Best Val AUC: {best_auc:.4f}")

# ── Final evaluation on best checkpoint ──────────────────────────────────────
print("\nRunning final evaluation on best checkpoint ...")
model.load_state_dict(torch.load('ember_lstm_best.pt'))
model.eval()

all_labels, all_probs = [], []
with torch.no_grad():
    for xb, yb in tqdm(test_loader, desc="Final Eval", leave=False):
        xb = xb.to(DEVICE, non_blocking=True)
        with torch.amp.autocast('cuda'):
            logits = model(xb)
        safe_probs = torch.sigmoid(torch.clamp(logits, -20, 20)).cpu().numpy()
        all_probs.extend(safe_probs)
        all_labels.extend(yb.numpy())

labels = np.array(all_labels)
probs  = np.nan_to_num(np.array(all_probs), nan=0.5, posinf=1.0, neginf=0.0)
preds  = (probs > 0.5).astype(int)

print(f"Final Test  acc={accuracy_score(labels, preds):.4f}  "
      f"auc={roc_auc_score(labels, probs):.4f}  "
      f"f1={f1_score(labels, preds, zero_division=0):.4f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
epochs_range = range(1, len(history) + 1)
tr_acc = [h['tr_acc'] for h in history]
va_acc = [h['va_acc'] for h in history]
tr_f1  = [h['tr_f1']  for h in history]
va_f1  = [h['va_f1']  for h in history]
tr_auc = [h['tr_auc'] for h in history]
va_auc = [h['va_auc'] for h in history]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("EmberLSTM Training Results", fontsize=14, fontweight='bold')

# Accuracy
axes[0].plot(epochs_range, tr_acc, 'b-o', label='Train')
axes[0].plot(epochs_range, va_acc, 'r-o', label='Val')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# F1 Score
axes[1].plot(epochs_range, tr_f1, 'b-o', label='Train')
axes[1].plot(epochs_range, va_f1, 'r-o', label='Val')
axes[1].set_title('F1 Score')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('F1')
axes[1].legend()
axes[1].grid(True)

# ROC Curve
fpr, tpr, _ = roc_curve(labels, probs)
axes[2].plot(fpr, tpr, 'b-', linewidth=2,
             label=f'AUC = {roc_auc_score(labels, probs):.4f}')
axes[2].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[2].set_title('ROC Curve (Test Set)')
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('ember_lstm_results.png', dpi=150, bbox_inches='tight')
print("Plots saved → ember_lstm_results.png")

# ── Export model ──────────────────────────────────────────────────────────────
torch.save(model.state_dict(), 'ember_lstm_weights.pt')
torch.save(model, 'ember_lstm_full.pt')

dummy    = torch.zeros(1, N_TIMESTEPS, N_FEATURES).to(DEVICE)
scripted = torch.jit.trace(model, dummy)
scripted.save('ember_lstm_scripted.pt')

print("\nModel exports:")
print("  ember_lstm_weights.pt   <- state dict  (resume training)")
print("  ember_lstm_full.pt      <- full model  (quick inference)")
print("  ember_lstm_scripted.pt  <- TorchScript (deployment)")