#!/usr/bin/env python3
# Test script to create a mismatch
import re

file_path = r'c:\LFM\workspace\website\src\data\research-experiments-generated.ts'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Change first REL-01 latticeSize from 512 to 256
content = content.replace('"latticeSize": 512', '"latticeSize": 256', 1)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Modified first latticeSize to 256 in website file')
