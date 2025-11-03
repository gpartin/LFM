#!/usr/bin/env python3
"""
Compare two directories recursively using SHA256 per-file and print a summary.
Usage:
  python tools/dirhash_compare.py <dirA> <dirB>
Outputs:
  - Added (in B not in A)
  - Removed (in A not in B)
  - Modified (present in both but different SHA256)
  - Unchanged count
"""
from __future__ import annotations
import sys
from pathlib import Path
import hashlib
from typing import Dict, Tuple


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def walk_hashes(root: Path) -> Dict[str, Tuple[int, str]]:
    mapping: Dict[str, Tuple[int, str]] = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            rel = str(path.relative_to(root)).replace('\\', '/')
            mapping[rel] = (path.stat().st_size, sha256_file(path))
    return mapping


def main() -> int:
    if len(sys.argv) != 3:
        print('Usage: python tools/dirhash_compare.py <dirA> <dirB>')
        return 2
    a = Path(sys.argv[1]).resolve()
    b = Path(sys.argv[2]).resolve()
    if not a.exists() or not b.exists():
        print('Both directories must exist')
        return 2
    ha = walk_hashes(a)
    hb = walk_hashes(b)

    keysA = set(ha.keys())
    keysB = set(hb.keys())
    added = sorted(keysB - keysA)
    removed = sorted(keysA - keysB)
    common = sorted(keysA & keysB)

    modified = []
    unchanged = 0
    for k in common:
        if ha[k][1] != hb[k][1]:
            modified.append(k)
        else:
            unchanged += 1

    print(f'Compare A={a} vs B={b}')
    print(f'  Added:   {len(added)}')
    print(f'  Removed: {len(removed)}')
    print(f'  Modified:{len(modified)}')
    print(f'  Unchanged:{unchanged}')

    if added:
        print('\nAdded files:')
        for k in added:
            size, h = hb[k]
            print(f'  + {k}  [{size} bytes]  {h}')
    if removed:
        print('\nRemoved files:')
        for k in removed:
            size, h = ha[k]
            print(f'  - {k}  [{size} bytes]  {h}')
    if modified:
        print('\nModified files:')
        for k in modified:
            sa, ha1 = ha[k]
            sb, hb1 = hb[k]
            print(f'  * {k}')
            print(f'      A: [{sa} bytes] {ha1}')
            print(f'      B: [{sb} bytes] {hb1}')

    return 0 if not (added or removed or modified) else 1


if __name__ == '__main__':
    raise SystemExit(main())
