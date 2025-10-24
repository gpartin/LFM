"""
LFM Phase-1 Runner — v2.1 (Auto-Discovery + Single Run Mode)
Automatically runs all Tier scripts or one specific script.
Assumes naming pattern:
    code/run_tierX_name.py  ↔  config/config_tierX_name.json
"""

import subprocess, sys, json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
CONFIG = ROOT / "config"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

BATCH_LOG = RESULTS / f"Phase1_Run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
BATCH_LOG.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# --- Helpers
# ---------------------------------------------------------------------------
def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None

def collect_metrics(summary_dir):
    path = Path(summary_dir)
    summaries = []
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
        for k, v in s["metrics"].items():
            if isinstance(v, (int, float)):
                metrics.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in metrics.items() if v}

def discover_scripts():
    """Find all run_tier*.py scripts in code/."""
    return sorted(CODE.glob("run_tier*.py"))

def find_config_for(script_path):
    """Find the matching config file by replacing 'run' with 'config'."""
    cfg_name = script_path.stem.replace("run", "config") + ".json"
    cfg_path = CONFIG / cfg_name
    return cfg_path if cfg_path.exists() else None

def run_script(script_path):
    """Run a single script and collect metrics if possible."""
    print(f"\n▶ Running {script_path.name} …")
    start_time = datetime.utcnow()

    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        status = "PASS ✅"
    except subprocess.CalledProcessError as e:
        status = f"FAIL ❌ (exit {e.returncode})"
    except Exception as ex:
        status = f"ERROR ❌ ({ex})"

    end_time = datetime.utcnow()

    tier_name = script_path.stem.split("_")[1].capitalize()
    result_dir = RESULTS / tier_name
    summaries = collect_metrics(result_dir)
    metrics_avg = flatten_metrics(summaries)

    return {
        "kernel": script_path.stem,
        "script": str(script_path),
        "status": status,
        "start_time": start_time.isoformat() + "Z",
        "end_time": end_time.isoformat() + "Z",
        "metrics": metrics_avg,
    }


# ---------------------------------------------------------------------------
# --- Main logic
# ---------------------------------------------------------------------------
if len(sys.argv) > 1:
    # Run a single script
    script_name = sys.argv[1]
    script_path = CODE / script_name
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        sys.exit(1)

    cfg_path = find_config_for(script_path)
    print(f"Using config: {cfg_path if cfg_path else '⚠ none found (default)'}")

    manifest = [run_script(script_path)]

else:
    # Run all scripts automatically
    scripts = discover_scripts()
    print(f"\n=== LFM Auto Batch Run ===")
    print(f"Discovered {len(scripts)} Tier scripts in /code")
    manifest = []

    for s in scripts:
        cfg_path = find_config_for(s)
        print(f"\n→ Matched config: {cfg_path if cfg_path else '⚠ none found'}")
        manifest.append(run_script(s))


# ---------------------------------------------------------------------------
# --- Manifest + summary
# ---------------------------------------------------------------------------
manifest_path = BATCH_LOG / "Phase1_summary.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("\n=== Phase-1 Summary ===")
for r in manifest:
    print(f"{r['kernel']:<35} {r['status']}")
print(f"\nFull manifest saved to: {manifest_path}")

print("\n=== Aggregate Metrics (averaged across variants) ===")
for r in manifest:
    m = r.get("metrics", {})
    if not m:
        continue
    print(f"\n[{r['kernel']}]")
    for k, v in m.items():
        print(f"  {k:<25} {v:.3e}")
