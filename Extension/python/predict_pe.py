#!/usr/bin/env python3
"""
PE Malware Detector - FINAL VERSION
Uses pefile (not LIEF) to match EMBER 2024 training schema exactly.
Requirements: pip install pefile torch numpy scikit-learn
"""
import argparse
import json
import hashlib
import math
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pefile
import pickle
import torch
import torch.nn as nn

# ── Config ────────────────────────────────────────────────────────────────────
METADATA_KEYS = {"sha256", "md5", "appeared", "label", "avclass",
                 "feature_version", "subset", "source"}
MAX_LIST_LEN = 256
TARGET_NF    = 64

# ── Model ─────────────────────────────────────────────────────────────────────
class EmberLSTM(nn.Module):
    def __init__(self, n_features: int, n_timesteps: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True,
                            dropout=0.3, bidirectional=True)
        self.attn = nn.Linear(512, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(512), nn.Dropout(0.3), nn.Linear(512, 128),
            nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        w = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1).unsqueeze(-1)
        context = (lstm_out * w).sum(dim=1)
        return self.classifier(context).squeeze(-1)


# ── Flattener — must match DAML.py exactly ────────────────────────────────────
def flatten_record(record, prefix=""):
    out = {}
    for k, v in record.items():
        fk = f"{prefix}.{k}" if prefix else k
        if fk in METADATA_KEYS:
            continue
        if isinstance(v, dict):
            out.update(flatten_record(v, fk))
        elif isinstance(v, list):
            for i, item in enumerate(v[:MAX_LIST_LEN]):
                if isinstance(item, dict):
                    out.update(flatten_record(item, f"{fk}.{i}"))
                elif isinstance(item, bool):
                    out[f"{fk}.{i}"] = float(item)
                elif isinstance(item, (int, float)):
                    val = float(item)
                    out[f"{fk}.{i}"] = 0.0 if not math.isfinite(val) else val
                # strings in lists are skipped — matches training behaviour
        elif isinstance(v, bool):
            out[fk] = float(v)
        elif isinstance(v, (int, float)):
            val = float(v)
            out[fk] = 0.0 if not math.isfinite(val) else val
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────
def entropy_bytes(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    probs  = [c / len(data) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


# ── Feature extraction using pefile ──────────────────────────────────────────
def extract_all_features(pe_path: str) -> dict:
    data = open(pe_path, "rb").read()
    sha256 = hashlib.sha256(data).hexdigest()
    md5    = hashlib.md5(data).hexdigest()

    try:
        pe = pefile.PE(data=data)
    except pefile.PEFormatError as e:
        raise RuntimeError(f"pefile could not parse PE: {e}")

    # ── histogram ─────────────────────────────────────────────────────────────
    hist = np.zeros(256, dtype=int)
    for b in data:
        hist[b] += 1
    hist = hist.tolist()

    # ── byteentropy (raw counts, NOT normalised — matches training) ───────────
    byteentropy = np.zeros(256, dtype=float)
    window, step = 1024, 512
    for i in range(0, max(1, len(data) - window + 1), step):
        block = data[i:i + window]
        e     = entropy_bytes(block)
        idx   = min(int(e * 32), 255)
        byteentropy[idx] += 1
    byteentropy = byteentropy.tolist()

    # ── strings ───────────────────────────────────────────────────────────────
    strings, cur = [], []
    for b in data:
        if 32 <= b < 127:
            cur.append(chr(b))
        else:
            if len(cur) >= 4:
                strings.append("".join(cur))
            cur = []
    if len(cur) >= 4:
        strings.append("".join(cur))

    str_lengths   = [len(s) for s in strings]
    printable_dist = np.zeros(96, dtype=int)
    for s in strings:
        for c in s:
            idx = ord(c) - 32
            if 0 <= idx < 96:
                printable_dist[idx] += 1

    all_text = " ".join(strings)

    # Exactly the 5 string_count keys the model was trained on
    string_counts = {
        "certificate": len(re.findall(r"certificate",  all_text, re.IGNORECASE)),
        "dos_msg":     len(re.findall(r"This program cannot be run in DOS mode", all_text)),
        "http://":     len(re.findall(r"http://",       all_text)),
        "service":     len(re.findall(r"service",       all_text, re.IGNORECASE)),
        "url":         len(re.findall(r"https?://[^\s]+", all_text)),
    }

    strings_bytes = bytearray()
    for s in strings:
        strings_bytes.extend(s.encode("ascii", errors="ignore"))

    strings_feat = {
        "numstrings":    len(strings),
        "avlength":      float(np.mean(str_lengths)) if str_lengths else 0.0,
        "printabledist": printable_dist.tolist(),
        "printables":    int(sum(printable_dist)),
        "string_counts": string_counts,
        "entropy":       entropy_bytes(bytes(strings_bytes)),
    }

    # ── general ───────────────────────────────────────────────────────────────
    general = {
        "entropy":     entropy_bytes(data),
        "is_pe":       1.0,
        "size":        len(data),
        "start_bytes": [int(b) for b in data[:4]] if len(data) >= 4 else [0,0,0,0],
    }

    # ── header.coff ───────────────────────────────────────────────────────────
    fh = pe.FILE_HEADER
    characteristics_list = []
    char_flags = {
        0x0001: "RELOCS_STRIPPED",        0x0002: "EXECUTABLE_IMAGE",
        0x0004: "LINE_NUMS_STRIPPED",     0x0008: "LOCAL_SYMS_STRIPPED",
        0x0010: "AGGRESIVE_WS_TRIM",      0x0020: "LARGE_ADDRESS_AWARE",
        0x0080: "BYTES_REVERSED_LO",      0x0100: "32BIT_MACHINE",
        0x0200: "DEBUG_STRIPPED",         0x0400: "REMOVABLE_RUN_FROM_SWAP",
        0x0800: "NET_RUN_FROM_SWAP",      0x1000: "SYSTEM",
        0x2000: "DLL",                    0x4000: "UP_SYSTEM_ONLY",
        0x8000: "BYTES_REVERSED_HI",
    }
    for bit, name in char_flags.items():
        if fh.Characteristics & bit:
            characteristics_list.append(name)

    coff = {
        "timestamp":               fh.TimeDateStamp,
        "machine":                 fh.Machine,
        "number_of_sections":      fh.NumberOfSections,
        "number_of_symbols":       fh.NumberOfSymbols,
        "pointer_to_symbol_table": fh.PointerToSymbolTable,
        "sizeof_optional_header":  fh.SizeOfOptionalHeader,
        "characteristics":         characteristics_list,
    }

    # ── header.dos ────────────────────────────────────────────────────────────
    dh = pe.DOS_HEADER
    dos = {
        "e_magic":    dh.e_magic,
        "e_cblp":     dh.e_cblp,
        "e_cp":       dh.e_cp,
        "e_crlc":     dh.e_crlc,
        "e_cparhdr":  dh.e_cparhdr,
        "e_minalloc": dh.e_minalloc,
        "e_maxalloc": dh.e_maxalloc,
        "e_ss":       dh.e_ss,
        "e_sp":       dh.e_sp,
        "e_csum":     dh.e_csum,
        "e_ip":       dh.e_ip,
        "e_cs":       dh.e_cs,
        "e_lfarlc":   dh.e_lfarlc,
        "e_ovno":     dh.e_ovno,
        "e_oemid":    dh.e_oemid,
        "e_oeminfo":  dh.e_oeminfo,
        "e_lfanew":   dh.e_lfanew,
    }

    # ── header.optional ───────────────────────────────────────────────────────
    optional = {}
    if hasattr(pe, "OPTIONAL_HEADER"):
        oh = pe.OPTIONAL_HEADER
        dll_chars = []
        dll_char_flags = {
            0x0001: "RESERVED_1",           0x0002: "RESERVED_2",
            0x0004: "RESERVED_4",           0x0008: "RESERVED_8",
            0x0010: "DYNAMIC_BASE",         0x0020: "FORCE_INTEGRITY",
            0x0040: "NX_COMPAT",            0x0080: "NO_ISOLATION",
            0x0100: "NO_SEH",               0x0200: "NO_BIND",
            0x0400: "APPCONTAINER",         0x0800: "WDM_DRIVER",
            0x1000: "GUARD_CF",             0x2000: "TERMINAL_SERVER_AWARE",
            0x4000: "HIGH_ENTROPY_VA",
        }
        for bit, name in dll_char_flags.items():
            if oh.DllCharacteristics & bit:
                dll_chars.append(name)

        subsystem_map = {
            0: "UNKNOWN", 1: "NATIVE", 2: "WINDOWS_GUI", 3: "WINDOWS_CUI",
            5: "OS2_CUI", 7: "POSIX_CUI", 9: "WINDOWS_CE_GUI",
            10: "EFI_APPLICATION", 11: "EFI_BOOT_SERVICE_DRIVER",
            12: "EFI_RUNTIME_DRIVER", 13: "EFI_ROM",
            14: "XBOX", 16: "WINDOWS_BOOT_APPLICATION",
        }
        subsystem_str = subsystem_map.get(oh.Subsystem, str(oh.Subsystem))

        optional = {
            "magic":                          oh.Magic,
            "major_linker_version":           oh.MajorLinkerVersion,
            "minor_linker_version":           oh.MinorLinkerVersion,
            "sizeof_code":                    oh.SizeOfCode,
            "sizeof_initialized_data":        oh.SizeOfInitializedData,
            "sizeof_uninitialized_data":      oh.SizeOfUninitializedData,
            "address_of_entrypoint":          oh.AddressOfEntryPoint,
            "base_of_code":                   oh.BaseOfCode,
            "base_of_data":                   getattr(oh, "BaseOfData", 0),
            "image_base":                     oh.ImageBase,
            "section_alignment":              oh.SectionAlignment,
            "major_operating_system_version": oh.MajorOperatingSystemVersion,
            "minor_operating_system_version": oh.MinorOperatingSystemVersion,
            "major_image_version":            oh.MajorImageVersion,
            "minor_image_version":            oh.MinorImageVersion,
            "major_subsystem_version":        oh.MajorSubsystemVersion,
            "minor_subsystem_version":        oh.MinorSubsystemVersion,
            "sizeof_image":                   oh.SizeOfImage,
            "sizeof_headers":                 oh.SizeOfHeaders,
            "checksum":                       oh.CheckSum,
            "sizeof_stack_reserve":           oh.SizeOfStackReserve,
            "sizeof_stack_commit":            oh.SizeOfStackCommit,
            "sizeof_heap_reserve":            oh.SizeOfHeapReserve,
            "sizeof_heap_commit":             oh.SizeOfHeapCommit,
            "number_of_rvas_and_sizes":       oh.NumberOfRvaAndSizes,
            "subsystem":                      subsystem_str,
            "dll_characteristics":            dll_chars,
        }

    header = {"coff": coff, "dos": dos, "optional": optional}

    # ── richheader — raw flat list matching training format ───────────────────
    # Training JSONs store: [id, count, id, count, ...] pairs (LIEF 0.9.0 format)
    # pefile gives us the same raw values via RICH_HEADER
    richheader = []
    if hasattr(pe, "RICH_HEADER"):
        rh = pe.RICH_HEADER
        if hasattr(rh, "values") and rh.values:
            for v in rh.values:
                richheader.append(int(v))
        if hasattr(rh, "key"):
            key = rh.key
            if isinstance(key, (bytes, bytearray)):
                import struct
                key = struct.unpack("<I", key[:4])[0]
            richheader.append(int(key))

    # ── sections — pad to 13 ──────────────────────────────────────────────────
    MAX_SECTIONS      = 13
    sections          = []
    entry_section_name = ""
    ep = optional.get("address_of_entrypoint", 0) if optional else 0

    for sec in pe.sections:
        try:
            sec_data = sec.get_data()
        except Exception:
            sec_data = b""

        char = sec.Characteristics
        props = []
        if char & 0x20000000: props.append("MEM_EXECUTE")
        if char & 0x40000000: props.append("MEM_READ")
        if char & 0x80000000: props.append("MEM_WRITE")
        if char & 0x00000020: props.append("CNT_CODE")
        if char & 0x00000040: props.append("CNT_INITIALIZED_DATA")
        if char & 0x00000080: props.append("CNT_UNINITIALIZED_DATA")
        if char & 0x02000000: props.append("MEM_DISCARDABLE")

        raw_size  = sec.SizeOfRawData
        virt_size = sec.Misc_VirtualSize
        virt_addr = sec.VirtualAddress
        name      = sec.Name.decode("utf-8", errors="replace").rstrip("\x00")

        sections.append({
            "name":       name,
            "size":       raw_size,
            "entropy":    entropy_bytes(sec_data),
            "vsize":      virt_size,
            "size_ratio": raw_size  / len(data) if len(data) > 0 else 0.0,
            "vsize_ratio": virt_size / raw_size  if raw_size  > 0 else 0.0,
            "props":      props,
        })

        if virt_addr <= ep < virt_addr + virt_size:
            entry_section_name = name

    while len(sections) < MAX_SECTIONS:
        sections.append({"name": "", "size": 0, "entropy": 0.0, "vsize": 0,
                         "size_ratio": 0.0, "vsize_ratio": 0.0, "props": []})

    # ── overlay ───────────────────────────────────────────────────────────────
    overlay = {"entropy": 0.0, "size": 0, "size_ratio": 0.0}
    try:
        if pe.sections:
            last_sec     = max(pe.sections,
                               key=lambda s: s.PointerToRawData + s.SizeOfRawData)
            overlay_start = last_sec.PointerToRawData + last_sec.SizeOfRawData
            if overlay_start < len(data):
                overlay_data         = data[overlay_start:]
                overlay["size"]      = len(overlay_data)
                overlay["entropy"]   = entropy_bytes(overlay_data)
                overlay["size_ratio"] = len(overlay_data) / len(data)
    except Exception:
        pass

    # ── imports ───────────────────────────────────────────────────────────────
    imports_dict = {}
    try:
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for imp in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = imp.dll.decode("utf-8", errors="replace") if imp.dll else "unknown"
                funcs    = []
                for entry in imp.imports:
                    if entry.name:
                        funcs.append(entry.name.decode("utf-8", errors="replace"))
                    elif entry.ordinal:
                        funcs.append(f"ord_{entry.ordinal}")
                    else:
                        funcs.append("unknown")
                imports_dict[dll_name] = funcs
    except Exception:
        pass

    # ── datadirectories — entry 0 is flags, entries 1-16 are named dirs ───────
    datadirs  = []
    dd_names  = ["EXPORT", "IMPORT", "RESOURCE", "EXCEPTION", "SECURITY",
                 "BASERELOC", "DEBUG", "COPYRIGHT", "GLOBALPTR", "TLS",
                 "LOAD_CONFIG", "BOUND_IMPORT", "IAT", "DELAY_IMPORT",
                 "COM_DESCRIPTOR", "RESERVED"]

    has_relocs   = 0
    actual_dirs  = {}
    try:
        if hasattr(pe, "OPTIONAL_HEADER") and hasattr(pe.OPTIONAL_HEADER, "DATA_DIRECTORY"):
            for i, dd in enumerate(pe.OPTIONAL_HEADER.DATA_DIRECTORY):
                if i < len(dd_names):
                    actual_dirs[dd_names[i]] = {"size": dd.Size,
                                                "virtual_address": dd.VirtualAddress}
                if dd_names[i] == "BASERELOC" and dd.Size > 0:
                    has_relocs = 1
    except Exception:
        pass

    datadirs.append({"has_relocs": has_relocs, "has_dynamic_relocs": 0})
    for name in dd_names:
        if name in actual_dirs:
            datadirs.append({"name": name,
                             "size": actual_dirs[name]["size"],
                             "virtual_address": actual_dirs[name]["virtual_address"]})
        else:
            datadirs.append({"name": name, "size": 0, "virtual_address": 0})

    # ── authenticode ──────────────────────────────────────────────────────────
    # pefile does not parse Authenticode deeply; we use sensible defaults.
    # The security directory presence tells us num_certs >= 1.
    authenticode = {
        "chain_max_depth":    0,
        "empty_program_name": 0.0,
        "latest_signing_time": 0.0,
        "no_countersigner":   0.0,
        "num_certs":          0,
        "parse_error":        0.0,
        "self_signed":        0.0,
        "signing_time_diff":  0.0,
    }
    try:
        sec_dir = actual_dirs.get("SECURITY", {})
        if sec_dir.get("size", 0) > 0:
            authenticode["num_certs"]          = 1
            authenticode["chain_max_depth"]    = 1
            authenticode["no_countersigner"]   = 1.0
            authenticode["empty_program_name"] = 1.0
    except Exception:
        authenticode["parse_error"] = 1.0

    pe.close()

    return {
        "sha256": sha256, "md5": md5, "appeared": "", "label": -1,
        "avclass": "unknown", "feature_version": 2, "subset": "test",
        "source": "pefile_extractor", "week_id": 0,
        "first_submission_date": 0, "last_analysis_date": 0,
        "authenticode": authenticode,
        "histogram":    hist,
        "byteentropy":  byteentropy,
        "strings":      strings_feat,
        "general":      general,
        "header":       header,
        "richheader":   richheader,
        "section":      {"entry": entry_section_name,
                         "sections": sections,
                         "overlay":  overlay},
        "imports":      imports_dict,
        "datadirectories": datadirs,
    }


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_pe(pe_path: str, model_dir: str) -> float:
    model_dir = Path(model_dir)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(model_dir / "feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    n_cols      = len(feature_cols)
    n_timesteps = max(1, math.ceil(n_cols / TARGET_NF))
    pad_to      = n_timesteps * TARGET_NF

    model = EmberLSTM(TARGET_NF, n_timesteps).to(device)
    model.load_state_dict(
        torch.load(model_dir / "ember_lstm_best.pt", map_location=device)
    )
    model.eval()

    print(f"Extracting features from {pe_path}...")
    record = extract_all_features(pe_path)
    flat   = flatten_record(record)
    print(f"Flattened features: {len(flat)}")

    # Load VT-only feature means (filled at inference since we can't extract them)
    vt_means = {}
    vt_means_path = model_dir / "vt_means.pkl"
    if vt_means_path.exists():
        with open(vt_means_path, "rb") as f:
            vt_means = pickle.load(f)

    features = np.zeros(n_cols, dtype=np.float32)
    missing  = []
    for i, col in enumerate(feature_cols):
        val = flat.get(col)
        if val is None:
            if col in vt_means:
                features[i] = float(vt_means[col])  # use training mean
            else:
                missing.append(col)
        elif np.isfinite(float(val)):
            features[i] = float(val)

    print(f"Features aligned: {n_cols - len(missing)}/{n_cols} present")
    if missing:
        print(f"Missing {len(missing)}: {sorted(missing)[:10]}")

    x_scaled = scaler.transform(features.reshape(1, -1))[0]
    if len(x_scaled) < pad_to:
        x_scaled = np.pad(x_scaled, (0, pad_to - len(x_scaled)))

    tensor = torch.tensor(
        x_scaled.reshape(n_timesteps, TARGET_NF), dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    conf = ("HIGH"   if prob > 0.9 else
            "MEDIUM" if prob > 0.7 else
            "LOW"    if prob > 0.5 else
            "BENIGN")

    print(f"\n{'='*50}")
    print(f"File:                {Path(pe_path).name}")
    print(f"Malware Probability: {prob:.4f}")
    print(f"Prediction:          {'MALWARE' if prob > 0.5 else 'BENIGN'}")
    print(f"Confidence:          {conf}")
    print(f"{'='*50}")
    return prob


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PE Malware Detector")
    parser.add_argument("--input",     required=True,
                        help="Path to PE file (.exe, .dll)")
    parser.add_argument("--model-dir", required=True,
                        help="Directory containing scaler.pkl, "
                             "feature_columns.pkl, ember_lstm_best.pt")
    args = parser.parse_args()
    predict_pe(args.input, args.model_dir)

if __name__ == "__main__":
    main()
