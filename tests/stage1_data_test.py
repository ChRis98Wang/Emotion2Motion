#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def run(cmd):
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        print(f"FAIL cmd: {' '.join(cmd)}")
        sys.exit(1)


def main():
    raw_dir = Path("data/raw")
    manifest = Path("data/manifests/dataset.yaml")
    parquet = Path("data/processed/clips.parquet")
    stats = Path("data/processed/stats.json")

    run(["python", "tools/gen_fake_csv.py", "--out_dir", str(raw_dir), "--num_demos", "3", "--fps_in", "120", "--duration", "8.0"])
    run(["python", "tools/make_manifest.py", "--raw_dir", str(raw_dir), "--out", str(manifest)])
    run(["python", "tools/convert_csv_to_parquet.py", "--manifest", str(manifest), "--out_parquet", str(parquet), "--out_stats", str(stats)])

    if not parquet.exists() or not stats.exists():
        print("FAIL outputs missing")
        sys.exit(1)

    df = pd.read_parquet(parquet)
    if len(df) == 0:
        print("FAIL no clips")
        sys.exit(1)

    sample = df.iloc[0]
    q = np.asarray(sample["q"])
    delta = np.asarray(sample["delta_q"])
    if q.dtype == object:
        q = np.stack(q, axis=0)
    if delta.dtype == object:
        delta = np.stack(delta, axis=0)
    q = q.astype(np.float32)
    delta = delta.astype(np.float32)

    if q.shape[0] != 240 or q.shape[1] != 4:
        print(f"FAIL shape {q.shape}")
        sys.exit(1)

    if not np.allclose(delta[0], 0.0, atol=1e-4):
        print("FAIL delta_q[0] not zero")
        sys.exit(1)

    stats_data = json.loads(stats.read_text())
    if len(stats_data.get("q_mean", [])) != 4:
        print("FAIL stats length")
        sys.exit(1)

    print("PASS stage1")


if __name__ == "__main__":
    main()
