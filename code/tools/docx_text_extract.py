#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Batch-extract text from .docx files without external dependencies.

- Recursively scans an input directory (default: ./documents) for .docx
- Extracts text from word/document.xml and writes .txt files mirroring structure
- Saves under docs/evidence/docx_text/<relative>.txt

Usage (from repo root):
  python tools/docx_text_extract.py [input_dir] [output_dir]

Examples:
  python tools/docx_text_extract.py documents docs/evidence/docx_text
  python tools/docx_text_extract.py . docs/evidence/docx_text

Note: This is a simple XML-stripper for docx; formatting is minimal by design.
"""

import sys
import os
import re
import zipfile
from html import unescape

DEFAULT_INPUT = os.path.join('.', 'documents')
DEFAULT_OUTPUT = os.path.join('docs', 'evidence', 'docx_text')

# Very simple XML tag stripper with paragraph/line break heuristics
TAG_RE = re.compile(r'<[^>]+>')

# Replace some known WordprocessingML tags with newlines/spaces before stripping
BREAK_MAP = [
    (re.compile(r'</w:p>'), '\n\n'),  # paragraph end
    (re.compile(r'<w:tab[^>]*/>'), '\t'),  # tabs
    (re.compile(r'<w:br[^>]*/>'), '\n'),   # line break
]

WHITESPACE_RE = re.compile(r'[\t \u00A0\u2000-\u200B]+')


def extract_docx_text(path: str) -> str:
    with zipfile.ZipFile(path) as zf:
        try:
            data = zf.read('word/document.xml').decode('utf-8', errors='ignore')
        except KeyError:
            return ''
    # Heuristic breaks
    for rx, repl in BREAK_MAP:
        data = rx.sub(repl, data)
    # Strip tags
    text = TAG_RE.sub('', data)
    # Unescape entities and normalize whitespace
    text = unescape(text)
    # Collapse multi-space but keep newlines
    lines = [WHITESPACE_RE.sub(' ', ln).strip() for ln in text.splitlines()]
    text = '\n'.join(ln for ln in lines if ln)
    return text.strip()


def main():
    in_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT

    if not os.path.isdir(in_dir):
        print(f"Input dir not found: {in_dir}")
        sys.exit(2)

    count = 0
    for root, _dirs, files in os.walk(in_dir):
        for fn in files:
            if not fn.lower().endswith('.docx'):
                continue
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, in_dir)
            rel_noext = os.path.splitext(rel)[0]
            dst = os.path.join(out_dir, rel_noext + '.txt')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                txt = extract_docx_text(src)
            except zipfile.BadZipFile:
                print(f"WARN: Not a valid .docx (zip) file: {src}")
                continue
            with open(dst, 'w', encoding='utf-8') as f:
                f.write(txt)
            print(f"Wrote: {dst}")
            count += 1
    if count == 0:
        print("No .docx files found.")
    else:
        print(f"Done. Extracted {count} file(s). Output in: {out_dir}")


if __name__ == '__main__':
    main()
