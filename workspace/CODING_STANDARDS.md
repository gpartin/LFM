# LFM Coding Standards

## File Encoding Rules

### **MANDATORY: Always specify UTF-8 encoding**

**Problem:** Windows defaults to `cp1252` encoding, which fails on Unicode characters (checkmarks ✓, arrows →, mathematical symbols ∇²).

**Solution:** ALWAYS specify `encoding='utf-8'` when opening files.

### File I/O Rules

```python
# ✓ CORRECT - Always specify encoding
with open(filename, 'r', encoding='utf-8') as f:
    content = f.read()

with open(filename, 'w', encoding='utf-8') as f:
    f.write(content)

with open(filename, 'a', encoding='utf-8') as f:
    f.write(content)

# ✗ WRONG - Platform-dependent encoding
with open(filename, 'r') as f:  # Fails on Windows with Unicode!
    content = f.read()
```

### JSON Files

```python
import json

# ✓ CORRECT
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# ✗ WRONG
with open('data.json', 'w') as f:
    json.dump(data, f)
```

### CSV Files

```python
import csv

# ✓ CORRECT
with open('data.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

with open('data.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)

# ✗ WRONG
with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
```

### Pandas DataFrames

```python
import pandas as pd

# ✓ CORRECT
df.to_csv('data.csv', encoding='utf-8', index=False)
df = pd.read_csv('data.csv', encoding='utf-8')

# ✗ WRONG
df.to_csv('data.csv', index=False)
```

### Path.write_text() / Path.read_text()

```python
from pathlib import Path

# ✓ CORRECT
Path('file.txt').write_text(content, encoding='utf-8')
content = Path('file.txt').read_text(encoding='utf-8')

# ✗ WRONG
Path('file.txt').write_text(content)  # Platform-dependent!
```

---

## Python Source File Encoding

### Always include UTF-8 declaration at top of file

```python
# -*- coding: utf-8 -*-
"""
Module docstring here.
"""

import numpy as np
```

**Why:** Ensures Python parser interprets source code as UTF-8, even on Windows.

**When needed:** 
- If file contains Unicode characters in comments/docstrings
- If file writes Unicode to stdout/stderr
- **Best practice:** Include in ALL Python files for consistency

---

## Print Statements and Logging

### Unicode-safe printing

```python
# ✓ CORRECT - Works on all platforms
print("Energy conservation: PASS")  # Use ASCII when possible
print("Algorithm 1: PASS")

# If you MUST use Unicode:
import sys
if sys.platform == 'win32':
    # Reconfigure stdout for UTF-8 on Windows
    sys.stdout.reconfigure(encoding='utf-8')

print("Energy conservation: ✓")  # Now safe

# ⚠️ CAUTION - May fail on Windows without reconfiguration
print("✓ Test passed")  # Can cause UnicodeEncodeError
```

### Logging

```python
import logging

# ✓ CORRECT - Configure logger for UTF-8
logging.basicConfig(
    filename='app.log',
    encoding='utf-8',  # Python 3.9+
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# For Python < 3.9:
handler = logging.FileHandler('app.log', encoding='utf-8')
logger.addHandler(handler)
```

---

## Common Unicode Characters - Use Carefully

### Safe to use in files (with UTF-8 encoding):
- Mathematical: `∇ ∂ ∫ ∑ π α β γ Δ`
- Arrows: `→ ← ↔ ⇒`
- Symbols: `≈ ≠ ≤ ≥ ±`
- Checkmarks: `✓ ✗`
- Bullets: `• ◦ ▪`

### Avoid in terminal output (unless stdout reconfigured):
- Use ASCII alternatives: `PASS`, `FAIL`, `->`, `!=`, `<=`, `>=`

---

## Git Configuration

Ensure git handles UTF-8 correctly:

```bash
git config --global core.quotepath false
git config --global i18n.commitEncoding utf-8
git config --global i18n.logOutputEncoding utf-8
```

---

## IDE/Editor Settings

### VS Code (`.vscode/settings.json`)

```json
{
  "files.encoding": "utf8",
  "files.autoGuessEncoding": false,
  "python.analysis.extraPaths": ["./src"],
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true
  }
}
```

---

## Testing

Always test on Windows to catch encoding issues:

```python
def test_unicode_file_io():
    """Test file I/O with Unicode characters."""
    from pathlib import Path
    
    test_content = "Energy: ✓ PASS\nGradient: ∇²E\nArrow: →"
    test_file = Path("test_unicode.txt")
    
    # Write with UTF-8
    test_file.write_text(test_content, encoding='utf-8')
    
    # Read with UTF-8
    read_content = test_file.read_text(encoding='utf-8')
    
    assert read_content == test_content
    test_file.unlink()  # Clean up
```

---

## Checklist for New Files

Before committing any Python file that does file I/O:

- [ ] All `open()` calls include `encoding='utf-8'`
- [ ] All `Path.write_text()` / `Path.read_text()` include `encoding='utf-8'`
- [ ] All `json.dump()` calls include `ensure_ascii=False` (optional but recommended)
- [ ] All `pd.read_csv()` / `pd.to_csv()` include `encoding='utf-8'`
- [ ] Source file includes `# -*- coding: utf-8 -*-` if using Unicode in source
- [ ] Tested on Windows (if applicable)

---

## Quick Reference Card

```python
# File I/O
open(path, 'r', encoding='utf-8')
open(path, 'w', encoding='utf-8')
Path(path).read_text(encoding='utf-8')
Path(path).write_text(text, encoding='utf-8')

# JSON
json.dump(data, f, ensure_ascii=False, indent=2)  # encoding='utf-8' on file handle

# CSV
csv.writer(f)  # encoding='utf-8' on file handle
pd.read_csv(path, encoding='utf-8')
pd.to_csv(path, encoding='utf-8')

# Logging
logging.basicConfig(filename='app.log', encoding='utf-8')

# Stdout (Windows fix)
sys.stdout.reconfigure(encoding='utf-8')
```

---

## Why This Matters

**Historical Context:**
- Python 3 uses Unicode (UTF-8) internally
- Windows still defaults to legacy encodings (cp1252, cp437)
- Mac/Linux default to UTF-8
- Result: Code that works on Mac/Linux fails on Windows

**LFM Project:**
- Scientific notation requires Unicode: `∇²E`, `∂²E/∂t²`
- Terminal output uses checkmarks: `✓`, `✗`
- Cross-platform compatibility is critical
- **One missed `encoding='utf-8'` breaks Windows builds**

**Solution:** Make UTF-8 encoding MANDATORY in all file operations.

---

## Enforcement

This is a **CRITICAL** requirement. All pull requests will be reviewed for encoding compliance.

**Automated check coming soon:** Pre-commit hook to detect missing `encoding=` parameters.
