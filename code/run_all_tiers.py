"""
LFM Phase-1 Batch Runner  — v1.1 (Aggregate Metrics Edition)
Executes Tier-1 → Tier-3 kernels sequentially and builds a consolidated summary.
"""

import subprocess, sys, json
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# --- Paths & setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

BATCH_LOG = RESULTS / f"Phase1_Run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
BATCH_LOG.mkdir(parents=True, exist_ok=True)

KERNELS = [
    ("Tier-1 Isotropy", CODE / "run_tier1_isotropy.py", ROOT / "results" / "Tier1" / "summary.json"),
    ("Tier-2 Pulse Propagation", CODE / "run_tier2_pulse_propagation.py", ROOT / "results" / "Tier2" / "Pulse"),
    ("Tier-2 Redshift", CODE / "run_tier2_redshift.py", ROOT / "results" / "Tier2" / "Redshift"),
    ("Tier-2 Curvature", CODE / "run_tier2_curvature_stability.py", ROOT / "results" / "Tier2" / "Curvature"),
    ("Tier-3 Entropy", CODE / "run_tier3_entropy_growth.py", ROOT / "results" / "Tier3" / "Entropy"),
]

results_manifest = []
print(f"\n=== LFM Phase-1 Validation Batch ===")
print(f"Results directory: {BATCH_LOG}\n")

# ---------------------------------------------------------------------------
# --- Helpers
# ---------------------------------------------------------------------------
def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None

def collect_metrics(summary_dir):
    """Locate and extract metrics from summary.json files under summary_dir."""
    summaries = []
    path = Path(summary_dir)
    if path.is_file() and path.name == "summary.json":
        summaries.append(read_json(path))
    elif path.is_dir():
        for f in path.rglob("summary.json"):
            summaries.append(read_json(f))
    return summaries

def flatten_metrics(summaries):
    metrics = {}
    for s in summaries or []:
        if not s or "metrics" not in s:
            continue
        for k,v in s["metrics"].items():
            if isinstance(v,(int,float)):
                metrics.setdefault(k, []).append(v)
    # Aggregate averages
    return {k: sum(v)/len(v) for k,v in metrics.items() if v}

# ---------------------------------------------------------------------------
# --- Sequential execution
# ---------------------------------------------------------------------------
for name, script, summary_path in KERNELS:
    print(f"\n▶ Running {name} …")
    start_time = datetime.utcnow()
    try:
        subprocess.run([sys.executable, str(script)], check=True)
        status = "PASS ✅"
    except subprocess.CalledProcessError as e:
        status = f"FAIL ❌ (exit {e.returncode})"
    except Exception as ex:
        status = f"ERROR ❌ ({ex})"
    end_time = datetime.utcnow()

    # Attempt to parse metrics
    summaries = collect_metrics(summary_path)
    metrics_avg = flatten_metrics(summaries)
    results_manifest.append({
        "kernel": name,
        "script": str(script),
        "status": status,
        "start_time": start_time.isoformat()+"Z",
        "end_time": end_time.isoformat()+"Z",
        "metrics": metrics_avg
    })
    print(f"→ {name} completed with status: {status}")

# ---------------------------------------------------------------------------
# --- Write batch manifest
# ---------------------------------------------------------------------------
manifest_path = BATCH_LOG / "Phase1_summary.json"
manifest_path.write_text(json.dumps(results_manifest, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------------
# --- Console summary
# ---------------------------------------------------------------------------
print("\n=== Phase-1 Batch Summary ===")
for r in results_manifest:
    print(f"{r['kernel']:<28} {r['status']}")
print(f"\nFull manifest saved to: {manifest_path}")

# ---------------------------------------------------------------------------
# --- Aggregate metrics table
# ---------------------------------------------------------------------------
print("\n=== Aggregate Metrics (means across variants) ===")
for r in results_manifest:
    m = r.get("metrics", {})
    if not m: 
        continue
    print(f"\n[{r['kernel']}]")
    for k,v in m.items():
        print(f"  {k:<25} {v:.3e}")
