#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(cmd):
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        print(f"FAIL cmd: {' '.join(cmd)}")
        sys.exit(1)


def main():
    parquet = Path("data/processed/clips.parquet")
    stats = Path("data/processed/stats.json")
    if not parquet.exists() or not stats.exists():
        run(["python", "tests/stage1_data_test.py"])

    save_dir = Path("data/tmp/checkpoints")
    run(
        [
            "python",
            "emg/train/train_flow.py",
            "--config",
            "configs/train_flow.yaml",
            "--num_steps",
            "120",
            "--batch_size",
            "4",
            "--save_dir",
            str(save_dir),
            "--overfit",
        ]
    )

    ckpt = save_dir / "flow_last.pt"
    if not ckpt.exists():
        print("FAIL checkpoint missing")
        sys.exit(1)

    log_path = save_dir / "train_log.json"
    if not log_path.exists():
        print("FAIL train_log missing")
        sys.exit(1)

    data = json.loads(log_path.read_text())
    losses = data.get("losses", [])
    if len(losses) < 100:
        print("FAIL not enough steps")
        sys.exit(1)

    first = sum(losses[:50]) / 50.0
    last = sum(losses[-50:]) / 50.0
    if not (last < first):
        print("FAIL loss not decreasing")
        sys.exit(1)

    print("PASS stage3")


if __name__ == "__main__":
    main()
