#!/usr/bin/env python3
import sys
import json
import os
import lief
import numpy as np

def extract_features(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        binary = lief.PE.parse(file_path)
        if not binary:
            raise ValueError("Not a valid PE file")
        
        features = []
        
        def to_float(val):
            if val is None:
                return 0.0
            if hasattr(val, 'value'):
                return float(val.value)
            try:
                f = float(val)
                return f if np.isfinite(f) else 0.0
            except:
                return 0.0
        
        # DOS Header (18 features)
        if hasattr(binary, 'dos_header') and binary.dos_header:
            dh = binary.dos_header
            features.append(to_float(getattr(dh, 'magic', 0)))
            features.append(to_float(getattr(dh, 'used_bytes_in_last_page', 0)))
            features.append(to_float(getattr(dh, 'file_size_in_pages', 0)))
            features.append(to_float(getattr(dh, 'numberof_relocation', 0)))
            features.append(to_float(getattr(dh, 'header_size_in_paragraphs', 0)))
            features.append(to_float(getattr(dh, 'minimum_extra_paragraphs', 0)))
            features.append(to_float(getattr(dh, 'maximum_extra_paragraphs', 0)))
            features.append(to_float(getattr(dh, 'initial_relative_ss', 0)))
            features.append(to_float(getattr(dh, 'initial_sp', 0)))
            features.append(to_float(getattr(dh, 'checksum', 0)))
            features.append(to_float(getattr(dh, 'initial_ip', 0)))
            features.append(to_float(getattr(dh, 'initial_relative_cs', 0)))
            features.append(to_float(getattr(dh, 'addressof_relocation_table', 0)))
            features.append(to_float(getattr(dh, 'overlay_number', 0)))
            features.append(to_float(getattr(dh, 'reserved', 0)))
            features.append(to_float(getattr(dh, 'oem_id', 0)))
            features.append(to_float(getattr(dh, 'oem_info', 0)))
            features.append(to_float(getattr(dh, 'addressof_new_exeheader', 0)))
        else:
            features.extend([0.0] * 18)
        
        # COFF Header (7 features)
        if hasattr(binary, 'header') and binary.header:
            h = binary.header
            features.append(to_float(getattr(h, 'machine', 0)))
            features.append(to_float(getattr(h, 'numberof_sections', 0)))
            features.append(to_float(getattr(h, 'time_date_stamps', 0)))
            features.append(to_float(getattr(h, 'pointerto_symbol_table', 0)))
            features.append(to_float(getattr(h, 'numberof_symbols', 0)))
            features.append(to_float(getattr(h, 'sizeof_optional_header', 0)))
            features.append(to_float(getattr(h, 'characteristics', 0)))
        else:
            features.extend([0.0] * 7)
        
        # Optional Header (29 features)
        if hasattr(binary, 'optional_header') and binary.optional_header:
            opt = binary.optional_header
            features.append(to_float(getattr(opt, 'magic', 0)))
            features.append(to_float(getattr(opt, 'major_linker_version', 0)))
            features.append(to_float(getattr(opt, 'minor_linker_version', 0)))
            features.append(to_float(getattr(opt, 'sizeof_code', 0)))
            features.append(to_float(getattr(opt, 'sizeof_initialized_data', 0)))
            features.append(to_float(getattr(opt, 'sizeof_uninitialized_data', 0)))
            features.append(to_float(getattr(opt, 'addressof_entrypoint', 0)))
            features.append(to_float(getattr(opt, 'baseof_code', 0)))
            features.append(to_float(getattr(opt, 'baseof_data', 0)))
            features.append(to_float(getattr(opt, 'imagebase', 0)))
            features.append(to_float(getattr(opt, 'section_alignment', 0)))
            features.append(to_float(getattr(opt, 'file_alignment', 0)))
            features.append(to_float(getattr(opt, 'major_operating_system_version', 0)))
            features.append(to_float(getattr(opt, 'minor_operating_system_version', 0)))
            features.append(to_float(getattr(opt, 'major_image_version', 0)))
            features.append(to_float(getattr(opt, 'minor_image_version', 0)))
            features.append(to_float(getattr(opt, 'major_subsystem_version', 0)))
            features.append(to_float(getattr(opt, 'minor_subsystem_version', 0)))
            features.append(to_float(getattr(opt, 'win32_version_value', 0)))
            features.append(to_float(getattr(opt, 'sizeof_image', 0)))
            features.append(to_float(getattr(opt, 'sizeof_headers', 0)))
            features.append(to_float(getattr(opt, 'checksum', 0)))
            features.append(to_float(getattr(opt, 'subsystem', 0)))
            features.append(to_float(getattr(opt, 'dll_characteristics', 0)))
            features.append(to_float(getattr(opt, 'sizeof_heap_reserve', 0)))
            features.append(to_float(getattr(opt, 'sizeof_heap_commit', 0)))
            features.append(to_float(getattr(opt, 'sizeof_stack_reserve', 0)))
            features.append(to_float(getattr(opt, 'sizeof_stack_commit', 0)))
            features.append(to_float(getattr(opt, 'loader_flags', 0)))
            features.append(to_float(getattr(opt, 'numberof_rva_and_size', 0)))
        else:
            features.extend([0.0] * 29)
        
        # Data Directory (16 entries * 2 = 32 features)
        if hasattr(binary, 'data_directories'):
            data_dirs = list(binary.data_directories) if binary.data_directories else []
            for i in range(16):
                if i < len(data_dirs):
                    dd = data_dirs[i]
                    features.append(to_float(getattr(dd, 'rva', 0)))
                    features.append(to_float(getattr(dd, 'size', 0)))
                else:
                    features.extend([0.0, 0.0])
        else:
            features.extend([0.0] * 32)
        
        # Sections (10 sections * 10 features = 100 features)
        if hasattr(binary, 'sections'):
            sections = list(binary.sections) if binary.sections else []
            for i in range(10):
                if i < len(sections):
                    sec = sections[i]
                    features.append(to_float(getattr(sec, 'virtual_address', 0)))
                    features.append(to_float(getattr(sec, 'virtual_size', 0)))
                    features.append(to_float(getattr(sec, 'sizeof_raw_data', 0)))
                    features.append(to_float(getattr(sec, 'pointerto_raw_data', 0)))
                    features.append(to_float(getattr(sec, 'pointerto_relocation', 0)))
                    features.append(to_float(getattr(sec, 'pointerto_line_numbers', 0)))
                    features.append(to_float(getattr(sec, 'numberof_relocations', 0)))
                    features.append(to_float(getattr(sec, 'numberof_line_numbers', 0)))
                    features.append(to_float(getattr(sec, 'characteristics', 0)))
                    features.append(to_float(getattr(sec, 'entropy', 0)))
                else:
                    features.extend([0.0] * 10)
        else:
            features.extend([0.0] * 100)
        
        # Imports (2 features)
        try:
            if hasattr(binary, 'imported_functions'):
                features.append(float(len(binary.imported_functions)))
            else:
                features.append(0.0)
        except:
            features.append(0.0)
            
        try:
            if hasattr(binary, 'imported_libraries'):
                features.append(float(len(binary.imported_libraries)))
            else:
                features.append(0.0)
        except:
            features.append(0.0)
        
        # Exports (1 feature)
        try:
            if hasattr(binary, 'exported_functions'):
                features.append(float(len(binary.exported_functions)))
            else:
                features.append(0.0)
        except:
            features.append(0.0)
        
        # TLS (3 features) - Check with hasattr
        if hasattr(binary, 'has_tls') and binary.has_tls and hasattr(binary, 'tls') and binary.tls:
            features.append(1.0)
            tls = binary.tls
            features.append(to_float(getattr(tls, 'addressof_callbacks', 0)))
            features.append(to_float(getattr(tls, 'characteristics', 0)))
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Resources (2 features) - Check with hasattr
        if hasattr(binary, 'has_resources') and binary.has_resources:
            features.append(1.0)
            try:
                if hasattr(binary, 'resources') and binary.resources:
                    features.append(float(len(binary.resources)))
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
        
        # Signature (1 feature) - Check with hasattr
        if hasattr(binary, 'has_signature') and binary.has_signature:
            features.append(1.0)
        else:
            features.append(0.0)
        
        # Pad to exactly 2568 features
        target = 2568
        while len(features) < target:
            features.append(0.0)
        features = features[:target]
        
        # Final sanitization
        features = [0.0 if not np.isfinite(f) else f for f in features]
        
        print(json.dumps(features))
        return 0
        
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {str(e)}"}), file=sys.stderr)
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: extract_features.py <file>"}), file=sys.stderr)
        sys.exit(1)
    sys.exit(extract_features(sys.argv[1]))