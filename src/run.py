"""Main pipeline entrypoint.
Usage:
    python -m src.run --split train|test --stage all|0|1|2|3|4
"""

import argparse
from src.config import load_config, set_seed


def run_pipeline(split: str = "test", stage: str = "all", config_path: str = "configs/default.yaml"):
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    print(f"Running pipeline for split='{split}' and stage='{stage}'")
    # Pipeline stages orchestration:
    # Stage 0: normalize
    # Stage 1: blocking
    # Stage 2: features
    # Stage 3: models / score
    # Stage 4: decide & write_outputs


def main():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Data split to run")
    parser.add_argument("--stage", choices=["all", "0", "1", "2", "3", "4"], default="all", help="Pipeline stage to execute")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()

    run_pipeline(split=args.split, stage=args.stage, config_path=args.config)


if __name__ == "__main__":
    main()
