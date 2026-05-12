import numpy as np
import pickle
from extract_features import file_to_raw_record, raw_record_to_flat, load_artifacts

feat_cols, scaler = load_artifacts('./memmap_cache')
col_index = {c: i for i, c in enumerate(feat_cols)}
n_cols = len(feat_cols)

def analyze_file(path, label):
    record = file_to_raw_record(path)
    flat = raw_record_to_flat(record)
    
    x_raw = np.zeros(n_cols, dtype=np.float32)
    for col, val in flat.items():
        idx = col_index.get(col)
        if idx is not None:
            x_raw[idx] = float(val)
    
    x_scaled = scaler.transform(x_raw.reshape(1, -1))[0].astype(np.float32)
    
    print(f'\n=== {label}: {path} ===')
    print(f'Features populated: {len(flat)} / {n_cols}')
    print(f'Raw min/max: {x_raw.min():.2f} / {x_raw.max():.2f}')
    print(f'Scaled min/max/mean: {x_scaled.min():.4f} / {x_scaled.max():.4f} / {x_scaled.mean():.4f}')
    print(f'Values > |3.0|: {(np.abs(x_scaled) > 3.0).sum()}')
    print(f'Values > |5.0|: {(np.abs(x_scaled) > 5.0).sum()}')
    print(f'Values > |10.0|: {(np.abs(x_scaled) > 10.0).sum()}')
    print(f'Values > |50.0|: {(np.abs(x_scaled) > 50.0).sum()}')
    
    extreme = []
    for i, val in enumerate(x_scaled):
        if abs(val) > 3.0:
            extreme.append((feat_cols[i], val, x_raw[i]))
    extreme.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f'\nTop 15 extreme scaled features:')
    print(f'{"Feature":<55} {"Scaled":>10} {"Raw":>12}')
    print('-' * 80)
    for col, sval, rval in extreme[:15]:
        print(f'{col:<55} {sval:>10.4f} {rval:>12.2f}')
    
    return x_raw, x_scaled

for path, label in [('hello64.exe', 'WINDOWS'), ('base.exe', 'LINUX')]:
    try:
        analyze_file(path, label)
    except Exception as e:
        print(f'{path}: {e}')



