# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

import json
import sys
from pathlib import Path
# Bootstrap to import path_utils from the workspace root
_WS_DIR = Path(__file__).resolve().parents[1]
if str(_WS_DIR) not in sys.path:
    sys.path.insert(0, str(_WS_DIR))
from path_utils import get_workspace_dir

test_id = sys.argv[1] if len(sys.argv) > 1 else 'QUAN-09'

WS = get_workspace_dir(__file__)
with open(WS / 'results' / 'test_metrics_history.json','r', encoding='utf-8') as f:
    d=json.load(f)
last=d.get(test_id,{}).get('runs',[])
print(f'{test_id}: Total runs: {len(last)}')
if last:
    last_run=last[-1]
    print(f'  CPU: {last_run.get("peak_cpu_percent", "MISSING")}%')
    print(f'  RAM: {last_run.get("peak_memory_mb", "MISSING")} MB')
    print(f'  GPU: {last_run.get("peak_gpu_memory_mb", "MISSING")} MB')
    print(f'  Runtime: {last_run.get("runtime_sec", "MISSING")} s')
    print(f'  Timestamp: {last_run.get("timestamp", "MISSING")}')
