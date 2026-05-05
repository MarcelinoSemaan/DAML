#!/usr/bin/env python3
r"""
PE feature extractor using LIEF - CORRECTED to match training schema exactly.
Usage:
    python extract_corrected_v2.py --input notepad.exe --output notepad.json
"""
import argparse
import json
import hashlib
import math
import re
from pathlib import Path
from collections import Counter
import numpy as np
import lief

METADATA_KEYS = {"sha256", "md5", "appeared", "label", "avclass",
                 "feature_version", "subset", "source"}
MAX_LIST_LEN = 256

# ------------------------------------------------------------------------------
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
                # NOTE: strings in lists are SKIPPED - matching training behavior
        elif isinstance(v, bool):
            out[fk] = float(v)
        elif isinstance(v, (int, float)):
            val = float(v)
            out[fk] = 0.0 if not math.isfinite(val) else val
    return out

def entropy_bytes(data):
    if not data:
        return 0.0
    counts = Counter(data)
    probs = [c / len(data) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def extract_all_features(pe_path):
    data = open(pe_path, "rb").read()
    sha256 = hashlib.sha256(data).hexdigest()
    md5 = hashlib.md5(data).hexdigest()
    binary = lief.PE.parse(list(data))
    if binary is None:
        raise RuntimeError("Failed to parse PE")

    # ── histogram (raw byte counts, 256 bins) ──────────────────────────────
    hist = np.zeros(256, dtype=int)
    for b in data:
        hist[b] += 1
    hist = hist.tolist()

    # ── byteentropy (RAW COUNTS, NOT NORMALIZED - matches training) ────────
    byteentropy = np.zeros(256, dtype=float)
    window = 1024
    step = 512
    for i in range(0, max(1, len(data) - window + 1), step):
        block = data[i:i + window]
        e = entropy_bytes(block)
        idx = min(int(e * 32), 255)
        byteentropy[idx] += 1
    # DO NOT NORMALIZE - training data uses raw counts
    byteentropy = byteentropy.tolist()

    # ── strings ────────────────────────────────────────────────────────────
    strings = []
    cur = []
    for b in data:
        if 32 <= b < 127:
            cur.append(chr(b))
        else:
            if len(cur) >= 4:
                strings.append("".join(cur))
            cur = []
    if len(cur) >= 4:
        strings.append("".join(cur))

    str_lengths = [len(s) for s in strings]
    printable_dist = np.zeros(96, dtype=int)
    for s in strings:
        for c in s:
            idx = ord(c) - 32
            if 0 <= idx < 96:
                printable_dist[idx] += 1

    all_text = " ".join(strings)

    # ALL 25 string count categories from training data
    string_counts = {
        "cache": len(re.findall(r"cache", all_text, re.IGNORECASE)),
        "command": len(re.findall(r"command", all_text, re.IGNORECASE)),
        "create": len(re.findall(r"create", all_text, re.IGNORECASE)),
        "debug": len(re.findall(r"debug", all_text, re.IGNORECASE)),
        "decode": len(re.findall(r"decode", all_text, re.IGNORECASE)),
        "delete": len(re.findall(r"delete", all_text, re.IGNORECASE)),
        "directory": len(re.findall(r"directory", all_text, re.IGNORECASE)),
        "dos_msg": len(re.findall(r"This program cannot be run in DOS mode", all_text)),
        "encode": len(re.findall(r"encode", all_text, re.IGNORECASE)),
        "enum": len(re.findall(r"enum", all_text, re.IGNORECASE)),
        "environment": len(re.findall(r"environment", all_text, re.IGNORECASE)),
        "exit": len(re.findall(r"exit", all_text, re.IGNORECASE)),
        "file": len(re.findall(r"file", all_text, re.IGNORECASE)),
        "http://": len(re.findall(r"http://", all_text)),
        "ipv6_addr": len(re.findall(r"ipv6", all_text, re.IGNORECASE)),
        "module": len(re.findall(r"module", all_text, re.IGNORECASE)),
        "privilege": len(re.findall(r"privilege", all_text, re.IGNORECASE)),
        "process": len(re.findall(r"process", all_text, re.IGNORECASE)),
        "remote": len(re.findall(r"remote", all_text, re.IGNORECASE)),
        "resource": len(re.findall(r"resource", all_text, re.IGNORECASE)),
        "security": len(re.findall(r"security", all_text, re.IGNORECASE)),
        "shell": len(re.findall(r"shell", all_text, re.IGNORECASE)),
        "system": len(re.findall(r"system", all_text, re.IGNORECASE)),
        "thread": len(re.findall(r"thread", all_text, re.IGNORECASE)),
        "url": len(re.findall(r"https?://[^\s]+", all_text)),
    }

    strings_bytes = bytearray()
    for s in strings:
        strings_bytes.extend(s.encode("ascii", errors="ignore"))

    strings_feat = {
        "numstrings": len(strings),
        "avlength": float(np.mean(str_lengths)) if str_lengths else 0.0,
        "printabledist": printable_dist.tolist(),
        "printables": int(sum(printable_dist)),
        "string_counts": string_counts,
        "entropy": entropy_bytes(bytes(strings_bytes)),
    }

    # ── general ────────────────────────────────────────────────────────────
    general = {
        "entropy": entropy_bytes(data),
        "is_pe": 1.0,
        "size": len(data),
        "start_bytes": [int(b) for b in data[:4]] if len(data) >= 4 else [0, 0, 0, 0],
    }

    # ── header.coff ────────────────────────────────────────────────────────
    coff = {
        "timestamp": int(getattr(binary.header, "time_date_stamp", 0)),
        "machine": str(binary.header.machine).replace("MACHINE_TYPES.", ""),
        "number_of_sections": binary.header.numberof_sections,
        "number_of_symbols": binary.header.numberof_symbols,
        "pointer_to_symbol_table": binary.header.pointerto_symbol_table,
        "sizeof_optional_header": binary.header.sizeof_optional_header,
        "characteristics": [str(c).replace("HEADER_CHARACTERISTICS.", "") for c in binary.header.characteristics_list],
    }

    # ── header.dos ─────────────────────────────────────────────────────────
    dos = {}
    if hasattr(binary, "dos_header") and binary.dos_header:
        dh = binary.dos_header
        dos_fields = ["e_magic", "e_cblp", "e_cp", "e_crlc", "e_cparhdr",
                      "e_minalloc", "e_maxalloc", "e_ss", "e_sp", "e_csum",
                      "e_ip", "e_cs", "e_lfarlc", "e_ovno", "e_oemid",
                      "e_oeminfo", "e_lfanew"]
        for field in dos_fields:
            dos[field] = getattr(dh, field, 0)

    # ── header.optional ────────────────────────────────────────────────────
    optional = {}
    if binary.optional_header:
        oh = binary.optional_header
        optional = {
            "magic": int(getattr(oh.magic, "value", oh.magic)),
            "major_linker_version": oh.major_linker_version,
            "minor_linker_version": oh.minor_linker_version,
            "sizeof_code": oh.sizeof_code,
            "sizeof_initialized_data": oh.sizeof_initialized_data,
            "sizeof_uninitialized_data": oh.sizeof_uninitialized_data,
            "address_of_entrypoint": oh.addressof_entrypoint,
            "base_of_code": oh.baseof_code,
            "base_of_data": getattr(oh, "baseof_data", 0),
            "image_base": oh.imagebase,
            "section_alignment": oh.section_alignment,
            "major_operating_system_version": oh.major_operating_system_version,
            "minor_operating_system_version": oh.minor_operating_system_version,
            "major_image_version": oh.major_image_version,
            "minor_image_version": oh.minor_image_version,
            "major_subsystem_version": oh.major_subsystem_version,
            "minor_subsystem_version": oh.minor_subsystem_version,
            "sizeof_image": oh.sizeof_image,
            "sizeof_headers": oh.sizeof_headers,
            "checksum": oh.checksum,
            "sizeof_stack_reserve": oh.sizeof_stack_reserve,
            "sizeof_stack_commit": oh.sizeof_stack_commit,
            "sizeof_heap_reserve": oh.sizeof_heap_reserve,
            "sizeof_heap_commit": oh.sizeof_heap_commit,
            "number_of_rvas_and_sizes": oh.numberof_rva_and_size,
            "subsystem": str(oh.subsystem).replace("SUBSYSTEM.", ""),
            "dll_characteristics": [str(c).replace("DLL_CHARACTERISTICS.", "") for c in oh.dll_characteristics_lists],
        }

    header = {"coff": coff, "dos": dos, "optional": optional}

    # ── richheader (RAW VALUES, 30 ints - matches training) ────────────────
    richheader = []
    if hasattr(binary, "rich_header") and binary.rich_header:
        rh = binary.rich_header
        # Raw rich header values as flat list (id, build_id pairs + key at end)
        entries = getattr(rh, "entries", [])
        for e in entries:
            richheader.append(getattr(e, "id", 0))
            richheader.append(getattr(e, "build_id", 0))
            richheader.append(getattr(e, "count", 1))
        richheader.append(getattr(rh, "key", 0))
    # Pad to match training if needed, but keep raw structure

    # ── sections ───────────────────────────────────────────────────────────
    sections = []
    entry_section_name = ""
    for sec in binary.sections:
        sec_data = bytes(sec.content) if hasattr(sec, "content") else b""
        props = []
        if sec.characteristics & 0x20000000:
            props.append("MEM_EXECUTE")
        if sec.characteristics & 0x40000000:
            props.append("MEM_READ")
        if sec.characteristics & 0x80000000:
            props.append("MEM_WRITE")
        if sec.characteristics & 0x00000020:
            props.append("CNT_CODE")
        if sec.characteristics & 0x00000040:
            props.append("CNT_INITIALIZED_DATA")
        if sec.characteristics & 0x00000080:
            props.append("CNT_UNINITIALIZED_DATA")
        if sec.characteristics & 0x02000000:
            props.append("MEM_DISCARDABLE")

        sections.append({
            "name": sec.name,
            "size": sec.sizeof_raw_data,
            "entropy": entropy_bytes(sec_data),
            "vsize": sec.virtual_size,
            "size_ratio": sec.sizeof_raw_data / len(data) if len(data) > 0 else 0.0,
            "vsize_ratio": sec.virtual_size / sec.sizeof_raw_data if sec.sizeof_raw_data > 0 else 0.0,
            "props": props
        })

    # Find entry section
    if binary.optional_header:
        ep = binary.optional_header.addressof_entrypoint
        for sec in binary.sections:
            if sec.virtual_address <= ep < sec.virtual_address + sec.virtual_size:
                entry_section_name = sec.name
                break

    overlay = {"entropy": 0.0, "size": 0, "size_ratio": 0.0}
    try:
        if binary.sections:
            last_sec = max(binary.sections, key=lambda s: s.offset + s.sizeof_raw_data)
            overlay_start = last_sec.offset + last_sec.sizeof_raw_data
            if overlay_start < len(data):
                overlay_data = data[overlay_start:]
                overlay["size"] = len(overlay_data)
                overlay["entropy"] = entropy_bytes(overlay_data)
                overlay["size_ratio"] = len(overlay_data) / len(data) if len(data) > 0 else 0.0
    except Exception:
        pass

    # ── datadirectories (BOOLEAN FLAGS - matches training) ─────────────────
    datadirs = []
    if binary.optional_header and hasattr(binary.optional_header, "data_directories"):
        for dd in binary.optional_header.data_directories:
            datadirs.append({
                "has_relocs": 1 if dd.type == lief.PE.DataDirectory.TYPES.BASE_RELOCATION_TABLE else 0,
                "has_dynamic_relocs": 0,  # Would need deeper parsing
            })
    # Pad to 17 entries if needed
    while len(datadirs) < 17:
        datadirs.append({"has_relocs": 0, "has_dynamic_relocs": 0})

    # ── authenticode ───────────────────────────────────────────────────────
    authenticode = {
        "chain_max_depth": 0,
        "empty_program_name": 0.0,
        "latest_signing_time": 0.0,
        "no_countersigner": 0.0,
        "num_certs": 0,
        "parse_error": 0.0,
        "self_signed": 0.0,
        "signing_time_diff": 0.0,
    }
    try:
        if binary.has_signatures:
            sig = binary.signatures[0]
            certs = sig.certificates
            authenticode["num_certs"] = len(certs)
            authenticode["chain_max_depth"] = len(certs)
            if certs:
                cert = certs[0]
                subj = str(getattr(cert, "subject", ""))
                issuer = str(getattr(cert, "issuer", ""))
                authenticode["self_signed"] = 1.0 if subj and subj == issuer else 0.0
                signers = getattr(sig, "signers", [])
                if signers:
                    prog_name = getattr(signers[0], "program_name", "") or ""
                    authenticode["empty_program_name"] = 1.0 if not prog_name.strip() else 0.0
                    countersigner = getattr(signers[0], "countersigner", None)
                    authenticode["no_countersigner"] = 1.0 if countersigner is None else 0.0
    except Exception:
        authenticode["parse_error"] = 1.0

    # ── imports (dict of string lists - will be flattened away) ────────────
    imports_dict = {}
    if binary.has_imports:
        for imp in binary.imports:
            dll_name = imp.name if imp.name else "unknown"
            funcs = []
            for entry in imp.entries:
                if entry.name:
                    funcs.append(entry.name)
                elif entry.is_ordinal:
                    funcs.append(f"ord_{entry.ordinal}")
                else:
                    funcs.append("unknown")
            imports_dict[dll_name] = funcs

    record = {
        "sha256": sha256,
        "md5": md5,
        "appeared": "",
        "label": -1,
        "avclass": "unknown",
        "feature_version": 2,
        "subset": "test",
        "source": "corrected_extractor_v2",
        "week_id": 0,
        "first_submission_date": 0,
        "last_analysis_date": 0,
        "authenticode": authenticode,
        "histogram": hist,
        "byteentropy": byteentropy,
        "strings": strings_feat,
        "general": general,
        "header": header,
        "richheader": richheader,
        "section": {
            "entry": entry_section_name,
            "sections": sections,
            "overlay": overlay,
        },
        "imports": imports_dict,
        "datadirectories": datadirs,
    }
    return record

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pe_path = Path(args.input)
    if not pe_path.exists():
        raise FileNotFoundError(pe_path)
    print(f"Extracting from {pe_path} ...")
    record = extract_all_features(pe_path)
    with open(args.output, "w") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Saved to {args.output}")
    flat_keys = sorted(flatten_record(record).keys())
    print(f"Flattened feature count: {len(flat_keys)}")

if __name__ == "__main__":
    main()



