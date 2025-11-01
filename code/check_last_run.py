import json
import sys

test_id = sys.argv[1] if len(sys.argv) > 1 else 'QUAN-09'

with open('c:/LFM/code/results/test_metrics_history.json','r') as f:
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
