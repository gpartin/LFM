#!/usr/bin/env python3
"""
Extended unit test for lfm_logger.py
Covers environment logging, JSONL structure, errors, and concurrent writes.
"""

import json
import tempfile
import threading
import time
from pathlib import Path
from lfm_logger import LFMLogger


def writer_thread(logger, idx):
    """Simulate a test writing mixed logs concurrently."""
    for i in range(10):
        logger.log(f"[Thread-{idx}] text message {i}")
        logger.log_json({"event": "thread_event", "thread": idx, "index": i})
        time.sleep(0.001)


def run_logger_test():
    tmp_dir = Path(tempfile.mkdtemp(prefix="lfm_logger_test_"))
    logger = LFMLogger(tmp_dir)
    logger.record_env(gpu_name="TestGPU", cuda_runtime=1234)
    logger.log("This is a main-thread test message.")
    logger.log_json({"event": "test_event", "value": 42})
    logger.error("Simulated error", Exception("FakeError"))

    # --- concurrent writes test ---
    threads = []
    for t in range(4):  # 4 threads writing simultaneously
        th = threading.Thread(target=writer_thread, args=(logger, t))
        threads.append(th)
        th.start()

    for th in threads:
        th.join()

    logger.close()

    text_log = tmp_dir / "session_log.txt"
    json_log = tmp_dir / "session_log.jsonl"
    assert text_log.exists(), "Text log missing."
    assert json_log.exists(), "JSON log missing."

    # --- basic checks ---
    text_content = text_log.read_text()
    assert "Simulated error" in text_content
    assert "Thread-0" in text_content

    json_lines = json_log.read_text().strip().splitlines()
    events = [json.loads(l) for l in json_lines]
    event_types = {e.get("event") for e in events}
    assert "environment" in event_types
    assert "test_event" in event_types
    assert "error" in event_types
    assert "thread_event" in event_types

    print(f"✅ Logger concurrent test passed ({len(json_lines)} JSON events). Logs in: {tmp_dir}")


if __name__ == "__main__":
    run_logger_test()
