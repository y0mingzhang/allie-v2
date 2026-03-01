#!/usr/bin/env python3
"""Monitor training and print status every N minutes. Run this instead of sleep."""

import subprocess
import sys
import time

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else "~/train_200M_v2b.log"
INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 300  # seconds

while True:
    result = subprocess.run(
        ["tail", "-1", LOG_FILE],
        capture_output=True,
        text=True,
    )
    line = result.stdout.strip()
    if "Step:" in line:
        print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)
    time.sleep(INTERVAL)
