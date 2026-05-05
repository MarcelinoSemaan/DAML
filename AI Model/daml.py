import json
import gc
import os
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve, precision_score

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = '/home/marcelino/Desktop/Data/Dataset/Combined-DB'
MEMMAP_DIR  = Path(DATA_DIR) / 'memmap_cache'  # Binary cache location
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = DEVICE.type
BATCH_SIZE  = 8192   # Increased - RAM is no longer the bottleneck
EPOCHS      = 15
N_WORKERS   = 4
TARGET_NF   = 64

METADATA_KEYS = {'sha256', 'md5', 'appeared', 'label', 'avclass',
                 'feature_version', 'subset', 'source'}
MAX_LIST_LEN  = 256

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

MEMMAP_DIR.mkdir(exist_ok=True)

# ── Record flattener ──────────────────────────────────────────────────────────
def flatten_record(record: dict, prefix: str = '') -> dict:
    out = {}
    for k, v in record.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if full_key in METADATA_KEYS:
            continue
        if isinstance(v, dict):
            out.update(flatten_record(v, full_key))
        elif isinstance(v, list):
            for i, item in enumerate(v[:MAX_LIST_LEN]):
                if isinstance(item, bool):
                    out[f"{full_key}.{i}"] = float(item)
                elif isinstance(item, (int, float)):
                    val = float(item)
                    out[f"{full_key}.{i}"] = 0.0 if not np.isfinite(val) else val
        elif isinstance(v, bool):
            out[full_key] = float(v)
        elif isinstance(v, (int, float)):
            val = float(v)
            out[full_key] = 0.0 if not np.isfinite(val) else val
    return out

# ── Memmap Creation ───────────────────────────────────────────────────────────
def discover_schema(data_dir):
    """Pass 1: Find feature columns from first labelled record"""
    print("Pass 1: discovering feature schema …")
    files = sorted(Path(data_dir).glob("*.jsonl"))
    train_files = [p for p in files if 'train' in p.name.lower()]
    
    for p in train_files:
        with p.open('r', encoding='utf-8') as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                    if int(rec.get('label', -1)) >= 0:
                        return sorted(flatten_record(rec).keys())
                except:
                    continue
    raise RuntimeError("No labelled records found for schema")

def convert_to_memmap(data_dir, feat_cols):
    """
    Converts JSONL → binary .dat files (unscaled).
    Creates: train_X.dat, train_y.dat, test_X.dat, test_y.dat
    """
    col_index = {c: i for i, c in enumerate(feat_cols)}
    n_cols = len(feat_cols)
    files = sorted(Path(data_dir).glob("*.jsonl"))
    train_files = [p for p in files if 'train' in p.name.lower()]
    test_files = [p for p in files if 'test' in p.name.lower()]
    
    def process_split(jsonl_files, split_name):
        # Count records
        print(f"\nPass 2: counting {split_name} …")
        n_records = 0
        for p in jsonl_files:
            with p.open('r', encoding='utf-8') as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        if int(json.loads(raw).get('label', -1)) >= 0:
                            n_records += 1
                    except:
                        continue
        
        print(f"  {split_name}: {n_records:,} records")
        
        # Create memmaps
        x_path = MEMMAP_DIR / f"{split_name}_X.dat"
        y_path = MEMMAP_DIR / f"{split_name}_y.dat"
        
        # Delete old if exists
        for p in [x_path, y_path]:
            if p.exists():
                p.unlink()
        
        X = np.memmap(x_path, dtype='float32', mode='w+', shape=(n_records, n_cols))
        y = np.memmap(y_path, dtype='int64', mode='w+', shape=(n_records,))
        
        # Fill
        print(f"Pass 3: filling {split_name} …")
        row = 0
        for p in jsonl_files:
            bad = 0
            with p.open('r', encoding='utf-8') as f:
                for raw in tqdm(f, desc=f"  {p.name}", leave=False):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except:
                        bad += 1
                        continue
                    lbl = int(rec.get('label', -1))
                    if lbl < 0:
                        continue
                    flat = flatten_record(rec)
                    for col, val in flat.items():
                        idx = col_index.get(col)
                        if idx is not None:
                            X[row, idx] = val
                    y[row] = int(lbl > 0)
                    row += 1
            if bad:
                print(f"    ⚠  {bad} bad lines in {p.name}")
        
        # Flush to disk
        X.flush()
        y.flush()
        
        # Save shape metadata
        with open(MEMMAP_DIR / f"{split_name}_meta.pkl", 'wb') as f:
            pickle.dump({'shape': (row, n_cols)}, f)
        
        return x_path, y_path, row
    
    train_x, train_y, n_train = process_split(train_files, 'train')
    test_x, test_y, n_test = process_split(test_files, 'test')
    
    return {
        'train': (train_x, train_y, n_train),
        'test': (test_x, test_y, n_test),
        'n_cols': n_cols
    }

# ── Incremental Scaler Fitting ────────────────────────────────────────────────
def fit_scaler_incrementally(X_path, shape, chunksize=5000):
    """
    Fits StandardScaler using partial_fit on chunks.
    Avoids loading entire dataset into RAM.
    """
    print("\nFitting StandardScaler incrementally on memmap …")
    scaler = StandardScaler()
    X = np.memmap(X_path, dtype='float32', mode='r', shape=shape)
    
    n_samples = shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)  # Randomize chunks for better statistics
    
    for start in tqdm(range(0, n_samples, chunksize), desc="Partial fit"):
        end = min(start + chunksize, n_samples)
        chunk = X[indices[start:end]]
        scaler.partial_fit(chunk)
    
    # Save scaler
    with open(MEMMAP_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    return scaler

def load_or_create_scaler(X_train_path, train_shape):
    scaler_path = MEMMAP_DIR / 'scaler.pkl'
    if scaler_path.exists():
        print("Loading existing scaler …")
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    return fit_scaler_incrementally(X_train_path, train_shape)

# ── Dataset ───────────────────────────────────────────────────────────────────
class MemmapDataset(Dataset):
    """
    Reads from memory-mapped files on-the-fly.
    Applies scaling and padding in __getitem__ (per-sample, minimal RAM).
    """
    def __init__(self, X_path, y_path, shape, scaler, n_timesteps, n_features, pad_to):
        self.X = np.memmap(X_path, dtype='float32', mode='r', shape=shape)
        self.y = np.memmap(y_path, dtype='int64', mode='r', shape=(shape[0],))
        self.scaler = scaler
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.pad_to = pad_to
        self.orig_features = shape[1]
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Disk read (2.8KB for 706 features) - OS caches automatically
        x_raw = self.X[idx].copy()
        
        # Scale (fitted on train, applied to both)
        x_scaled = self.scaler.transform(x_raw.reshape(1, -1))[0]
        
        # Pad if needed (e.g., 706 → 768)
        if len(x_scaled) < self.pad_to:
            x_scaled = np.pad(x_scaled, (0, self.pad_to - len(x_scaled)), constant_values=0.0)
        
        # Reshape for LSTM
        x_reshaped = x_scaled.reshape(self.n_timesteps, self.n_features)
        
        return (torch.tensor(x_reshaped, dtype=torch.float32),
                torch.tensor(float(self.y[idx]), dtype=torch.float32))

# ── Model ─────────────────────────────────────────────────────────────────────
class EmberLSTM(nn.Module):
    def __init__(self, n_features: int, n_timesteps: int):
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

# ── Training Utilities ────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, amp_scaler, train=True, desc=""):
    nan_batches = 0
    model.train() if train else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    
    bar = tqdm(loader, desc=desc, leave=False)
    for xb, yb in bar:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        
        with torch.amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == 'cuda')):
            logits = model(xb)
            logits = torch.clamp(logits, -20.0, 20.0)
            loss = criterion(logits, yb)
        
        if not torch.isfinite(loss):
            nan_batches += 1
            if train:
                optimizer.zero_grad(set_to_none=True)
            continue
        
        if train:
            optimizer.zero_grad(set_to_none=True)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(yb.cpu().numpy())
        bar.set_postfix(loss=f"{loss.item():.4f}")
    
    if nan_batches:
        print(f"  ⚠  {nan_batches} NaN/Inf batches skipped")
    
    if not all_labels:
        return float('nan'), 0.0, 0.0, 0.0
    
    labels = np.array(all_labels)
    probs = np.nan_to_num(np.array(all_probs), nan=0.5, posinf=1.0, neginf=0.0)
    preds = (probs > 0.5).astype(int)
    
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0
    
    return (total_loss / max(len(loader) - nan_batches, 1),
            accuracy_score(labels, preds),
            auc,
            f1_score(labels, preds, zero_division=0))

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check if memmaps exist
    required = ['train_X.dat', 'train_y.dat', 'test_X.dat', 'test_y.dat',
                'train_meta.pkl', 'test_meta.pkl']
    cache_exists = all((MEMMAP_DIR / f).exists() for f in required)
    
    if not cache_exists:
        print("Creating memory-mapped cache (one-time operation) …")
        feat_cols = discover_schema(DATA_DIR)
        meta = convert_to_memmap(DATA_DIR, feat_cols)
    else:
        print("Using existing memory-mapped cache.")
        # Load metadata
        with open(MEMMAP_DIR / 'train_meta.pkl', 'rb') as f:
            train_meta = pickle.load(f)
        with open(MEMMAP_DIR / 'test_meta.pkl', 'rb') as f:
            test_meta = pickle.load(f)
        meta = {
            'train': (MEMMAP_DIR / 'train_X.dat', MEMMAP_DIR / 'train_y.dat', train_meta['shape'][0]),
            'test': (MEMMAP_DIR / 'test_X.dat', MEMMAP_DIR / 'test_y.dat', test_meta['shape'][0]),
            'n_cols': train_meta['shape'][1]
        }
    
    n_cols = meta['n_cols']
    train_shape = (meta['train'][2], n_cols)
    test_shape = (meta['test'][2], n_cols)
    
    print(f"\nDataset loaded from disk cache:")
    print(f"  Train: {train_shape[0]:,} samples × {n_cols} features")
    print(f"  Test:  {test_shape[0]:,} samples × {n_cols} features")
    
    # Feature reshape logic (with ceil fix)
    N_TIMESTEPS = max(1, math.ceil(n_cols / TARGET_NF))
    N_FEATURES = TARGET_NF
    pad_to = N_TIMESTEPS * N_FEATURES
    
    if pad_to > n_cols:
        print(f"Padding: {n_cols} → {pad_to} ({N_TIMESTEPS}×{N_FEATURES})")
    
    # Fit scaler incrementally (RAM-safe)
    scaler = load_or_create_scaler(meta['train'][0], train_shape)
    
    # Create datasets
    train_ds = MemmapDataset(meta['train'][0], meta['train'][1], train_shape,
                             scaler, N_TIMESTEPS, N_FEATURES, pad_to)
    test_ds = MemmapDataset(meta['test'][0], meta['test'][1], test_shape,
                            scaler, N_TIMESTEPS, N_FEATURES, pad_to)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=N_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=N_WORKERS, pin_memory=True)
    
    print(f"Batches: Train={len(train_loader)}, Test={len(test_loader)}")
    
    # Model
    model = EmberLSTM(N_FEATURES, N_TIMESTEPS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    
    # Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=1e-4,
                           steps_per_epoch=len(train_loader), epochs=EPOCHS,
                           pct_start=0.3, div_factor=10, final_div_factor=100)
    amp_scaler = torch.amp.GradScaler(device=DEVICE_TYPE, enabled=(DEVICE_TYPE == 'cuda'))
    
    # Training
    history = []
    best_auc = 0.0
    CKPT_PATH = 'ember_lstm_best.pt'
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{EPOCHS} ──")
        
        tr = run_epoch(model, train_loader, criterion, optimizer, amp_scaler, True, 
                       f"Epoch {epoch}/{EPOCHS} Train")
        scheduler.step()
        
        with torch.no_grad():
            va = run_epoch(model, test_loader, criterion, optimizer, amp_scaler, False,
                           f"Epoch {epoch}/{EPOCHS} Val")
        
        print(f"  Train  loss={tr[0]:.4f}  acc={tr[1]:.4f}  auc={tr[2]:.4f}  f1={tr[3]:.4f}")
        print(f"  Val    loss={va[0]:.4f}  acc={va[1]:.4f}  auc={va[2]:.4f}  f1={va[3]:.4f}")
        
        history.append(dict(tr_loss=tr[0], tr_acc=tr[1], tr_auc=tr[2], tr_f1=tr[3],
                           va_loss=va[0], va_acc=va[1], va_auc=va[2], va_f1=va[3]))
        
        if va[2] > best_auc:
            best_auc = va[2]
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  ✓ Saved (AUC={best_auc:.4f})")
    
    print(f"\nDone. Best Val AUC: {best_auc:.4f}")
    
    # Final eval
    print("\nRunning final evaluation …")
    if os.path.exists(CKPT_PATH):
        model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Final Eval", leave=False):
            xb = xb.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE_TYPE, enabled=(DEVICE_TYPE == 'cuda')):
                logits = model(xb)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(yb.numpy())
    
    labels = np.array(all_labels)
    probs = np.nan_to_num(np.array(all_probs), nan=0.5, posinf=1.0, neginf=0.0)
    preds = (probs > 0.5).astype(int)
    print(f"Final Test  acc={accuracy_score(labels, preds):.4f}  "
          f"auc={roc_auc_score(labels, probs):.4f}  "
          f"f1={f1_score(labels, preds, zero_division=0):.4f}")
    
    # Plots
    epochs_range = range(1, len(history) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("EmberLSTM Training Results (Memmap)", fontsize=14, fontweight='bold')
    
    for ax, tk, vk, title, ylabel in [
        (axes[0], 'tr_acc', 'va_acc', 'Accuracy', 'Accuracy'),
        (axes[1], 'tr_f1', 'va_f1', 'F1 Score', 'F1'),
    ]:
        ax.plot(epochs_range, [h[tk] for h in history], 'b-o', label='Train')
        ax.plot(epochs_range, [h[vk] for h in history], 'r-o', label='Val')
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel); ax.legend(); ax.grid(True)
    
    fpr, tpr, _ = roc_curve(labels, probs)
    final_auc = roc_auc_score(labels, probs)
    axes[2].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {final_auc:.4f}')
    axes[2].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[2].set_title('ROC Curve (Test Set)'); axes[2].set_xlabel('FPR')
    axes[2].set_ylabel('TPR'); axes[2].legend(); axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('ember_lstm_results.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved → ember_lstm_results.png")
    
    # Export
    torch.save(model.state_dict(), 'ember_lstm_weights.pt')
    dummy = torch.zeros(1, N_TIMESTEPS, N_FEATURES).to(DEVICE)
    scripted = torch.jit.trace(model, dummy)
    scripted.save('ember_lstm_scripted.pt')
    print("Exports: ember_lstm_weights.pt, ember_lstm_scripted.pt")

    # ── Plots ─────────────────────────────────────────────────────────────────────
from sklearn.metrics import precision_score  # Add this import at top of file

epochs_range = range(1, len(history) + 1)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("EmberLSTM Training Results", fontsize=16, fontweight='bold')

# Plot 1: Accuracy over epochs
ax = axes[0, 0]
ax.plot(epochs_range, [h['tr_acc'] for h in history], 'b-o', label='Train', markersize=4)
ax.plot(epochs_range, [h['va_acc'] for h in history], 'r-o', label='Val', markersize=4)
ax.set_title('Accuracy per Epoch')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

# Plot 2: F1 over epochs
ax = axes[0, 1]
ax.plot(epochs_range, [h['tr_f1'] for h in history], 'b-o', label='Train', markersize=4)
ax.plot(epochs_range, [h['va_f1'] for h in history], 'r-o', label='Val', markersize=4)
ax.set_title('F1 Score per Epoch')
ax.set_xlabel('Epoch'); ax.set_ylabel('F1')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

# Plot 3: ROC Curve
ax = axes[1, 0]
fpr, tpr, _ = roc_curve(labels, probs)
final_auc = roc_auc_score(labels, probs)
ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'AUC = {final_auc:.4f}')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
ax.fill_between(fpr, tpr, alpha=0.15, color='blue')
ax.set_title('ROC Curve (Test Set)')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)

# Plot 4: Final Metrics Histogram (Aggregate)
ax = axes[1, 1]
final_acc = accuracy_score(labels, preds)
final_f1 = f1_score(labels, preds, zero_division=0)
final_prec = precision_score(labels, preds, zero_division=0)

metrics = ['Accuracy', 'Precision', 'F1 Score']
values = [final_acc, final_prec, final_f1]
colors = ['#2ecc71', '#3498db', '#e74c3c']

bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.set_title('Final Test Metrics (Aggregate)')

# Add value labels on top of bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.annotate(f'{val:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add horizontal reference lines
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Random (0.5)')
ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.3, label='Excellent (0.9)')
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
plt.savefig('ember_lstm_results.png', dpi=150, bbox_inches='tight')
print("Plots saved → ember_lstm_results.png")