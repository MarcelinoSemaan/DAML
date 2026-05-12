#!/usr/bin/env python3
"""
compile_c.py
============
Compiles a single C file to a .exe using MinGW-w64.

Usage:
    python compile_c.py hello.c
    python compile_c.py hello.c --output myapp.exe
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# Common compiler names across platforms
COMPILER_CANDIDATES = [
    "gcc",                       # Windows MinGW / most POSIX
    "x86_64-w64-mingw32-gcc",    # Linux/macOS cross-compiler
    "i686-w64-mingw32-gcc",      # 32-bit cross-compiler
    "mingw32-gcc",               # Older MinGW
]


def find_compiler() -> str:
    """Locate a suitable C compiler in PATH."""
    for name in COMPILER_CANDIDATES:
        path = shutil.which(name)
        if path:
            print(f"[INFO] Found compiler: {path}")
            return path

    sys.exit(
        "[ERROR] No C compiler found in PATH.\n\n"
        "Windows (MinGW-w64):\n"
        "  1. Download from https://github.com/niXman/mingw-builds-binaries/releases\n"
        "     or install via MSYS2: https://www.msys2.org/\n"
        "  2. Add the bin/ folder to your PATH (e.g., C:\\mingw64\\bin)\n"
        "  3. Restart your terminal\n\n"
        "Linux:\n"
        "  sudo apt install mingw-w64       # Debian/Ubuntu\n"
        "  sudo dnf install mingw64-gcc     # Fedora\n"
        "  sudo pacman -S mingw-w64-gcc     # Arch\n\n"
        "macOS:\n"
        "  brew install mingw-w64"
    )


def compile(source: Path, output: Path) -> Path:
    """Compile source C file to output executable."""
    if not source.exists():
        sys.exit(f"[ERROR] Source file not found: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)

    compiler = find_compiler()
    cmd = [compiler, str(source), "-o", str(output)]

    print(f"[CMD] {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Compilation failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    if result.stderr:
        print(f"[WARN] {result.stderr.strip()}")

    print(f"[OK] Compiled: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Compile C to executable")
    parser.add_argument("source", help="C source file (.c)")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output executable path (default: dist/<basename>.exe)"
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()

    if args.output:
        output = Path(args.output).resolve()
    else:
        output = source.parent / "dist" / f"{source.stem}.exe"

    compile(source, output)


if __name__ == "__main__":
    main()