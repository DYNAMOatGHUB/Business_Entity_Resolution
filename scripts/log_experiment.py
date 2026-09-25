"""Log an experiment or leaderboard submission to reports/experiments.csv."""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def log_experiment(description: str, oof_f05: float, loco_f05: float = None, lb_score: float = None, out_csv: str = "reports/experiments.csv"):
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "description": description,
        "oof_f05": oof_f05,
        "loco_f05": loco_f05,
        "lb_score": lb_score,
    }

    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(path, index=False)
    print(f"Logged experiment to {path}")


def main():
    parser = argparse.ArgumentParser(description="Log experiment")
    parser.add_argument("--desc", required=True, help="Experiment description")
    parser.add_argument("--oof", type=float, required=True, help="OOF F0.5 score")
    parser.add_argument("--loco", type=float, default=None, help="LOCO F0.5 score")
    parser.add_argument("--lb", type=float, default=None, help="Public leaderboard score")
    args = parser.parse_args()

    log_experiment(args.desc, args.oof, args.loco, args.lb)


if __name__ == "__main__":
    main()
