#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd


def _moving_average(x, window):
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.vstack([np.convolve(x[:, j], kernel, mode="same") for j in range(x.shape[1])]).T


def synth_traj(num_joints, fps, duration, seed, amp_scale=1.0):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0 / fps, dtype=np.float32)
    T = t.shape[0]

    q = np.zeros((T, num_joints), dtype=np.float32)
    for j in range(num_joints):
        freq = 0.15 + 0.05 * j + rng.uniform(0.0, 0.1)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = (0.6 + rng.uniform(0.0, 0.3)) * amp_scale
        sinus = amp * np.sin(2.0 * np.pi * freq * t + phase)

        key_times = np.linspace(0.0, duration, 6, dtype=np.float32)
        key_vals = rng.uniform(-0.6, 0.6, size=(6,))
        piece = np.interp(t, key_times, key_vals)

        q[:, j] = sinus + 0.4 * piece

    q = _moving_average(q, window=max(3, int(0.03 * fps)))
    q = np.clip(q, -1.2, 1.2)
    return t, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_demos", type=int, default=3)
    ap.add_argument("--num_joints", type=int, default=4)
    ap.add_argument("--fps_in", type=int, default=120)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = ["greet", "idle", "point", "celebrate", "apologize"]

    for i in range(args.num_demos):
        task = tasks[i % len(tasks)]
        amp_scale = 0.7 if (i % 2 == 0) else 1.3
        t, q = synth_traj(args.num_joints, args.fps_in, args.duration, args.seed + i, amp_scale=amp_scale)

        data = {"time": t}
        for j in range(args.num_joints):
            data[f"J{j+1}"] = q[:, j]

        demo_id = f"demo_{i+1:04d}_{task}"
        csv_path = out_dir / f"{demo_id}.csv"
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
