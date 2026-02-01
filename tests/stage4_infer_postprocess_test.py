#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

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
        run(["python", "tests/stage3_train_smoke_test.py"])

    out_dir = Path("data/tmp/generated")
    run(["python", "emg/infer/generate_clip.py", "--config", "configs/infer.yaml", "--out_dir", str(out_dir)])

    csv_path = out_dir / "q_clip.csv"
    if not csv_path.exists():
        print("FAIL q_clip missing")
        sys.exit(1)

    q = pd.read_csv(csv_path).to_numpy(dtype=np.float32)
    if q.shape != (240, 4):
        print(f"FAIL shape {q.shape}")
        sys.exit(1)

    cfg = yaml.safe_load(Path("configs/infer.yaml").read_text())
    q_min = np.array(cfg["postprocess"]["q_min"], dtype=np.float32)
    q_max = np.array(cfg["postprocess"]["q_max"], dtype=np.float32)
    dq_max = np.array(cfg["postprocess"]["dq_max"], dtype=np.float32)

    if np.any(q < q_min - 1e-5) or np.any(q > q_max + 1e-5):
        print("FAIL joint limits")
        sys.exit(1)

    dq = np.diff(q, axis=0) * cfg.get("fps", 60)
    if np.any(np.abs(dq) > dq_max * 1.05):
        print("FAIL velocity limits")
        sys.exit(1)

    max_jump = float(np.max(np.abs(q[1:] - q[:-1])))
    if max_jump > 0.5:
        print("FAIL continuity")
        sys.exit(1)

    print("PASS stage4")


if __name__ == "__main__":
    main()
