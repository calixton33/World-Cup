"""Run each completed pipeline stage in order, without notebook dependencies."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("data", "features", "backtest", "final", "readme")


def run(script: str, arguments: list[str]) -> None:
    """Run one pipeline command and stop immediately on failure."""
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *arguments], cwd=ROOT, check=True)


def main() -> None:
    """Support both the full rebuild and cumulative milestone verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--through", choices=STAGES, default="readme")
    parser.add_argument("--as-of", default=datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat())
    parser.add_argument("--train-start-year", type=int, default=1990)
    parser.add_argument("--calibration-years", type=int, default=4)
    args = parser.parse_args()
    if not args.skip_download:
        run("download_data.py", [])
    date_args = ["--as-of", args.as_of]
    model_args = ["--train-start-year", str(args.train_start_year),
                  "--calibration-years", str(args.calibration_years)]
    commands = [
        ("audit_data.py", date_args), ("build_features.py", date_args),
        ("run_backtest.py", model_args), ("train_final.py", []), ("write_readme.py", []),
    ]
    for script, arguments in commands[:STAGES.index(args.through) + 1]:
        run(script, arguments)


if __name__ == "__main__":
    main()
