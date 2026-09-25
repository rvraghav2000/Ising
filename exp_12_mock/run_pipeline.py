# run_pipeline.py - runs everything and saves output to a log file

# --- path bootstrap (added by repo reorganisation) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.join(_HERE, '..', 'common'))
_sys.path.insert(0, _os.path.join(_HERE, '..', 'exp_12_real'))
# --- end path bootstrap ---

import sys
import os
from datetime import datetime

# writes to terminal and log file at the same time
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
    print(f"\ndone, output saved to: {log_path}\n")


if __name__ == "__main__":
    main()
