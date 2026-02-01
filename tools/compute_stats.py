#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    def _to_array(x):
        arr = np.asarray(x)
        if arr.dtype == object:
            arr = np.stack(arr, axis=0)
        return arr.astype(np.float32)

    q_list = [_to_array(q) for q in df["q"].tolist()]
    d_list = [_to_array(d) for d in df["delta_q"].tolist()]

    q_cat = np.concatenate(q_list, axis=0)
    d_cat = np.concatenate(d_list, axis=0)

    stats = {
        "q_mean": q_cat.mean(axis=0).astype(np.float32).tolist(),
        "q_std": (q_cat.std(axis=0) + 1e-6).astype(np.float32).tolist(),
        "delta_mean": d_cat.mean(axis=0).astype(np.float32).tolist(),
        "delta_std": (d_cat.std(axis=0) + 1e-6).astype(np.float32).tolist(),
    }

    Path(args.out).write_text(json.dumps(stats, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
