#!/usr/bin/env python3
"""Extract embedded GLSL sources from AssetRipper yaml shader exports.

The compressedBlob in each .asset contains the shader's compiled programs for
multiple GPU targets, including readable GLSL (HLSLcc cross-compiled output).
We pull out every #ifdef VERTEX / FRAGMENT program block.
"""
import re
import sys
import pathlib

def extract(blob_hex: str):
    data = bytes.fromhex(blob_hex)
    text = data.decode('latin-1')  # byte-faithful
    # GLSL programs start with "#ifdef VERTEX" or "#ifdef FRAGMENT" (possibly after a #version)
    # and run to the next null-ish terminator or end marker.
    programs = []
    # find all candidate starts
    for m in re.finditer(r'#ifdef (VERTEX|FRAGMENT|GEOMETRY)', text):
        start = m.start()
        # program text runs until a long run of non-printable bytes
        i = start
        while i < len(text):
            ch = text[i]
            if ord(ch) < 9 or (13 < ord(ch) < 32):
                # allow short runs of control chars? GLSL text is clean; stop at first non-printable
                break
            i += 1
        prog = text[start:i]
        if len(prog) > 50:  # filter noise
            programs.append((m.group(1), prog))
    return programs

def main():
    path = sys.argv[1]
    content = pathlib.Path(path).read_text()
    m = re.search(r'compressedBlob: ([0-9a-fA-F]+)', content)
    if not m:
        print("no compressedBlob found")
        return 1
    programs = extract(m.group(1))
    print(f"// {len(programs)} GLSL programs in {path}")
    for kind, prog in programs:
        print(f"\n// ===== {kind} =====")
        print(prog)

if __name__ == '__main__':
    sys.exit(main())
