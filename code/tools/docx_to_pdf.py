#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Batch-convert .docx files to PDF using docx2pdf.

Usage (from repo root):
  python tools/docx_to_pdf.py documents docs/evidence/pdf

Requires: docx2pdf (pip install docx2pdf)
"""
import sys
import os
from docx2pdf import convert

def main():
    in_dir = sys.argv[1] if len(sys.argv) > 1 else 'documents'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'docs/evidence/pdf'
    os.makedirs(out_dir, exist_ok=True)
    print(f"Converting all .docx in {in_dir} to PDF in {out_dir}...")
    convert(in_dir, out_dir)
    print("Done.")

if __name__ == '__main__':
    main()
