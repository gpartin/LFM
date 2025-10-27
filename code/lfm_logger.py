#!/usr/bin/env python3
"""
lfm_logger.py — Unified logging system for all LFM tiers
Outputs both text and JSONL logs for each test or suite run.
"""

import json
from datetime import datetime
from pathlib import Path
import platform

class LFMLogger:
    """Dual-format logger: human text + structured JSONL."""
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.text_log = self.base_dir / "session_log.txt"
        self.json_log = self.base_dir / "session_log.jsonl"
        self._write_header()

    def _write_header(self):
        header = f"=== LFM Log — {datetime.utcnow().isoformat()}Z ===\n"
        if not self.text_log.exists():
            self.text_log.write_text(header, encoding="utf-8")
        else:
            with open(self.text_log, "a", encoding="utf-8") as f:
                f.write("\n" + header)

    def log(self, msg):
        """Append timestamped text entry."""
        line = f"[{datetime.utcnow().isoformat()}Z] {msg}\n"
        with open(self.text_log, "a", encoding="utf-8") as f:
            f.write(line)

    def log_json(self, obj):
        """Append structured JSONL event (auto-sanitized)."""
        def sanitize(o):
            if isinstance(o, (bool, int, float, str)) or o is None:
                return o
            if hasattr(o, "item"):  # numpy/cupy scalar
                return o.item()
            return str(o)

        clean = {k: sanitize(v) for k, v in obj.items()}
        clean["timestamp"] = datetime.utcnow().isoformat() + "Z"
        with open(self.json_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(clean) + "\n")

    def record_env(self, gpu_name="Unknown", cuda_runtime=0):
        """Log basic environment info once per session."""
        env = {
            "event": "environment",
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "gpu": gpu_name,
            "cuda_runtime": cuda_runtime
        }
        self.log_json(env)
        self.log(f"Environment: {env}")

    def error(self, msg, err=None):
        """Record an error with traceback text."""
        entry = {"event": "error", "message": msg}
        if err:
            entry["exception"] = str(err)
        self.log_json(entry)
        self.log(f"[ERROR] {msg}")

    def close(self):
        """Flush and mark log end."""
        line = f"--- End of Log {datetime.utcnow().isoformat()}Z ---\n"
        with open(self.text_log, "a", encoding="utf-8") as f:
            f.write(line)
        with open(self.json_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "end", "timestamp": datetime.utcnow().isoformat() + "Z"}) + "\n")
