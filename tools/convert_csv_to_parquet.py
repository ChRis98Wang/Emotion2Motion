#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yaml


def _find_time_col(columns):
    for name in columns:
        lname = name.lower()
        if lname in ("t", "time"):
            return name
    return None


def _infer_joint_cols(columns, time_col):
    cols = [c for c in columns if c != time_col]
    return cols


def _lowpass_filter(q, fps, cutoff_hz):
    if cutoff_hz is None or cutoff_hz <= 0:
        return q
    try:
        from scipy.signal import butter, filtfilt
    except Exception:
        warnings.warn("scipy not available; skipping lowpass")
        return q

    nyq = 0.5 * fps
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(3, normal_cutoff, btype="low", analog=False)
    q_f = filtfilt(b, a, q, axis=0)
    return q_f.astype(np.float32)


def _resample(q, t_in, fps_out):
    t0 = float(t_in[0])
    t1 = float(t_in[-1])
    if t1 <= t0:
        raise ValueError("time span too small")
    dt = 1.0 / fps_out
    t_out = np.arange(t0, t1 + 1e-9, dt, dtype=np.float32)
    q_out = np.zeros((t_out.shape[0], q.shape[1]), dtype=np.float32)
    for j in range(q.shape[1]):
        q_out[:, j] = np.interp(t_out, t_in, q[:, j]).astype(np.float32)
    return q_out


def _slice_clips(q, clip_len, stride):
    clips = []
    for start in range(0, q.shape[0] - clip_len + 1, stride):
        clips.append(q[start : start + clip_len])
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--out_stats", required=True)
    args = ap.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text())

    fps_out = int(manifest.get("fps_out", 60))
    clip_sec = float(manifest.get("clip_sec", 4.0))
    stride_sec = float(manifest.get("stride_sec", 2.0))
    lowpass_hz = manifest.get("lowpass_hz", None)
    use_delta_q = bool(manifest.get("use_delta_q", True))
    joints_expected = int(manifest.get("joints_expected", 4))
    angle_unit = manifest.get("angle_unit", "rad")

    clip_len = int(round(clip_sec * fps_out))
    stride = int(round(stride_sec * fps_out))

    rows = []
    all_q = []
    all_dq = []

    for item in manifest.get("items", []):
        demo_id = item["demo_id"]
        csv_path = Path(item["csv_path"])
        if not csv_path.exists():
            raise FileNotFoundError(str(csv_path))

        df = pd.read_csv(csv_path)
        time_col = _find_time_col(df.columns)
        if time_col is not None:
            t_in = df[time_col].to_numpy(dtype=np.float32)
        else:
            fps_in = item.get("fps_in", manifest.get("fps_in", None))
            if fps_in is None:
                raise ValueError(f"missing fps_in for {demo_id} (no time column)")
            t_in = np.arange(df.shape[0], dtype=np.float32) / float(fps_in)

        joint_names = item.get("joint_names")
        if joint_names is None:
            joint_names = _infer_joint_cols(df.columns, time_col)
        if len(joint_names) != joints_expected:
            raise ValueError(f"expected {joints_expected} joints, got {len(joint_names)} in {demo_id}")

        q = df[joint_names].to_numpy(dtype=np.float32)
        if angle_unit == "deg":
            q = np.deg2rad(q).astype(np.float32)

        if lowpass_hz:
            q = _lowpass_filter(q, fps=(1.0 / (t_in[1] - t_in[0])), cutoff_hz=lowpass_hz)

        q = _resample(q, t_in, fps_out)

        clips = _slice_clips(q, clip_len, stride)
        if not clips:
            warnings.warn(f"no clips produced for {demo_id}")
            continue

        for idx, q_clip in enumerate(clips):
            q_clip = q_clip.astype(np.float32)
            delta_q = np.zeros_like(q_clip)
            delta_q[1:] = q_clip[1:] - q_clip[:-1]
            dq = np.zeros_like(q_clip)
            dq[1:] = (q_clip[1:] - q_clip[:-1]) * fps_out

            rows.append(
                {
                    "clip_id": f"{demo_id}_clip_{idx:04d}",
                    "demo_id": demo_id,
                    "fps": fps_out,
                    "T": clip_len,
                    "joint_names": list(joint_names),
                    "q": q_clip.tolist(),
                    "delta_q": delta_q.tolist(),
                    "dq": dq.tolist(),
                    "q0": q_clip[0].tolist(),
                    "emotion_valence": float(item.get("emotion", {}).get("valence", 0.0)),
                    "emotion_arousal": float(item.get("emotion", {}).get("arousal", 0.0)),
                    "emotion_text": item.get("emotion", {}).get("text", ""),
                    "task_type": item.get("context", {}).get("task_type", "idle"),
                    "context_json": json.dumps(item.get("context", {}), ensure_ascii=True),
                    "meta_json": json.dumps(
                        {
                            "source_fps": float(1.0 / (t_in[1] - t_in[0])),
                            "angle_unit": angle_unit,
                            "time_col": time_col,
                            "joint_cols": list(joint_names),
                        },
                        ensure_ascii=True,
                    ),
                }
            )

            all_q.append(q_clip)
            all_dq.append(delta_q)

    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_parquet(out_parquet, index=False)

    if not all_q:
        raise RuntimeError("no clips found; stats not computed")

    q_cat = np.concatenate(all_q, axis=0)
    d_cat = np.concatenate(all_dq, axis=0)

    stats = {
        "q_mean": q_cat.mean(axis=0).astype(np.float32).tolist(),
        "q_std": (q_cat.std(axis=0) + 1e-6).astype(np.float32).tolist(),
        "delta_mean": d_cat.mean(axis=0).astype(np.float32).tolist(),
        "delta_std": (d_cat.std(axis=0) + 1e-6).astype(np.float32).tolist(),
        "fps": fps_out,
    }

    out_stats = Path(args.out_stats)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, indent=2))

    print(f"wrote {out_parquet}")
    print(f"wrote {out_stats}")


if __name__ == "__main__":
    main()
