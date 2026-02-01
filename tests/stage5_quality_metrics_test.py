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
    ckpt = Path("data/tmp/checkpoints/flow_last.pt")
    if not ckpt.exists():
        run(["python", "tests/stage4_infer_postprocess_test.py"])

    report = Path("data/tmp/report.json")
    run(
        [
            "python",
            "emg/train/eval_basic.py",
            "--ckpt",
            str(ckpt),
            "--parquet",
            "data/processed/clips.parquet",
            "--stats",
            "data/processed/stats.json",
            "--infer_config",
            "configs/infer.yaml",
            "--report",
            str(report),
        ]
    )

    if not report.exists():
        print("FAIL report missing")
        sys.exit(1)

    data = json.loads(report.read_text())
    if data.get("limit_violation_rate", 1.0) != 0.0:
        print("FAIL violation_rate")
        sys.exit(1)

    smooth = data.get("smoothness", 999.0)
    if smooth > 10.0:
        print("FAIL smoothness too high")
        sys.exit(1)

    sens = data.get("condition_sensitivity", 0.0)
    if sens <= 1e-4:
        print("FAIL sensitivity too low")
        sys.exit(1)

    print("PASS stage5")


if __name__ == "__main__":
    main()
