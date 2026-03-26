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
import json
import pandas as pd
from pathlib import Path

def load_data(data_dir):
import json
import numpy as np
import pandas as pd
from pathlib import Path
import thrember

# ── Ember/Thrember feature dimensions (adjust if using a different version) ──
EMBER_NUM_FEATURES = 2381   # default for ember2018; change to 2351 for older builds

def _memmap_features(data_dir: Path, subset: str):
    """
    Memory-map the raw .dat files instead of loading them fully into RAM.
    Falls back to thrember.read_vectorized_features if the .dat files
    aren't in the expected location.
    """
    X_path = data_dir / f"X_{subset}.dat"
    y_path = data_dir / f"y_{subset}.dat"

    if X_path.exists() and y_path.exists():
        # mmap_mode='r' → file stays on disk, OS pages in only what's touched
        X = np.memmap(str(X_path), dtype=np.float32, mode="r").reshape(-1, EMBER_NUM_FEATURES)
        y = np.memmap(str(y_path), dtype=np.float32, mode="r")
        print(f"  [{subset}] memory-mapped  X={X.shape}  y={y.shape}")
        return X, y
    else:
        # fallback — will load fully into RAM
        print(f"  [{subset}] .dat not found, falling back to thrember loader (full RAM load)")
        return thrember.read_vectorized_features(str(data_dir), subset=subset)


def load_data(data_dir):
    data_dir = Path(data_dir)

    # ── 1. Stream JSONL files in chunks ──────────────────────────────────────
    files       = sorted(data_dir.glob("*.jsonl"))
    total_files = len(files)
    file_count  = line_count = bad_lines = 0
    chunks  = []
    records = []
    CHUNK   = 50_000          # rows per flush; lower if records are wide

    for p in files:
        file_count += 1
        print(f"Processing file {file_count}/{total_files}: {p}")

        with p.open("r", encoding="utf-8") as f:
            for i, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    records.append(json.loads(raw))
                    line_count += 1
                except json.JSONDecodeError:
                    bad_lines += 1

                if len(records) >= CHUNK:
                    chunks.append(pd.DataFrame.from_records(records))
                    records.clear()

        print(f"  -> lines read: {i}, valid: {line_count}, bad: {bad_lines}")

    if records:
        chunks.append(pd.DataFrame.from_records(records))
        records.clear()

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    del chunks, records                  # free staging memory NOW
    print(f"Finished JSONL. Files: {total_files}, Records: {len(df)}, Bad lines: {bad_lines}")

    # ── 2. Memory-map train features ─────────────────────────────────────────
    print("Loading vectorized features (train)…")
    X_train, y_train = _memmap_features(data_dir, "train")

    # ── 3. Memory-map test features (this was the OOM trigger) ───────────────
    print("Loading vectorized features (test)…")
    X_test, y_test = _memmap_features(data_dir, "test")

    return df, X_train, y_train, X_test, y_test


# ── Entry point ───────────────────────────────────────────────────────────────
DATA_DIR = "/home/marcelino/Desktop/Data/Dataset/Vectorised-Data"

df, X_train, y_train, X_test, y_test = load_data(DATA_DIR)

print(f"JSONL DataFrame : {df.shape}")
print(f"Train features  : {X_train.shape}  labels: {y_train.shape}")
print(f"Test  features  : {X_test.shape}   labels: {y_test.shape}")
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

train_loader = DataLoader(EmberDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=N_WORKERS, pin_memory=True)
test_loader  = DataLoader(EmberDataset(X_test, y_test),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=N_WORKERS, pin_memory=True)

print(f"Train batches: {len(train_loader)}  Test batches: {len(test_loader)}")

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
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
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

        total_loss += loss.item()
        safe_probs = torch.sigmoid(torch.clamp(logits, -20, 20)).detach().cpu().numpy()
        all_probs.extend(safe_probs)
        all_labels.extend(yb.cpu().numpy())
        bar.set_postfix(loss=f"{loss.item():.4f}")

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
scaler    = torch.amp.GradScaler('cuda')
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