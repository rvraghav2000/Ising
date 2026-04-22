#!/usr/bin/env python3
"""
run_pipeline.py
===============
Master runner: executes both the Quantum Portfolio Optimizer and the
Sensitivity Analysis, saving all terminal output to a timestamped log file.
"""

import sys
import os
from datetime import datetime

# ── Tee: write to both terminal and a log file simultaneously ──
class Tee:
    def __init__(self, file):
        self._file    = file
        self._stdout  = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


def main():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join("logs", f"pipeline_{timestamp}.txt")

    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(log_file)

        print(f"Pipeline run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file            : {log_path}")
        print("=" * 108)

        # ── 1. Main Optimizer ──
        print()
        print("STAGE 1 / 2  —  Quantum Portfolio Optimizer")
        print("=" * 108)
        import quantum_portfolio_optimizer as qpo
        qpo.main()

        # ── 2. Sensitivity Analysis ──
        print()
        print("STAGE 2 / 2  —  Sensitivity Analysis")
        print("=" * 108)
        import sensitivity_analysis as sa
        sa.run_analysis()

        print()
        print("=" * 108)
        print(f"Pipeline complete. Log saved to: {log_path}")
        print("=" * 108)

    # Restore stdout after the 'with' block closes the file
    sys.stdout = sys.__stdout__
    print(f"\n✅  Full output saved to: {log_path}\n")


if __name__ == "__main__":
    main()
