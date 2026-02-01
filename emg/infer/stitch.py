#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def stitch_clips(clips, overlap):
    if not clips:
        return np.zeros((0, 0), dtype=np.float32)
    out = clips[0]
    for nxt in clips[1:]:
        if overlap <= 0:
            out = np.concatenate([out, nxt], axis=0)
            continue
        ov = min(overlap, out.shape[0], nxt.shape[0])
        fade = np.linspace(0.0, 1.0, ov, dtype=np.float32)[:, None]
        blended = out[-ov:] * (1.0 - fade) + nxt[:ov] * fade
        out = np.concatenate([out[:-ov], blended, nxt[ov:]], axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--overlap", type=int, default=30)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    csvs = sorted(in_dir.glob("*.csv"))
    clips = [pd.read_csv(p).to_numpy(dtype=np.float32) for p in csvs]
    out = stitch_clips(clips, args.overlap)
    pd.DataFrame(out).to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
