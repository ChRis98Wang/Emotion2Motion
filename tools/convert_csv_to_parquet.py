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
    return [c for c in columns if c != time_col]


def _flatten_numeric(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        out = []
        for x in value:
            out.extend(_flatten_numeric(x))
        return out
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)]
    return []


def _context_lookup(context, key):
    cur = context
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
            continue
        return None
    return cur


def _infer_context_key_dims(items, context_num_keys):
    dims = {}
    for key in context_num_keys:
        dim = 1
        for item in items:
            ctx = item.get("context", {})
            val = _context_lookup(ctx, key)
            flat = _flatten_numeric(val)
            if flat:
                dim = max(1, len(flat))
                break
        dims[key] = dim
    return dims


def _build_context_vector(context, context_num_keys, context_key_dims):
    vec = []
    for key in context_num_keys:
        dim = int(context_key_dims.get(key, 1))
        val = _context_lookup(context, key)
        flat = _flatten_numeric(val)
        if not flat:
            flat = [0.0] * dim
        elif len(flat) < dim:
            flat = flat + [0.0] * (dim - len(flat))
        elif len(flat) > dim:
            flat = flat[:dim]
        vec.extend(flat)
    return [float(x) for x in vec]


def _estimate_fps(t_in):
    dt = np.diff(t_in)
    dt = dt[dt > 1e-9]
    if dt.size == 0:
        raise ValueError("invalid or duplicate-only timestamps")
    return float(1.0 / np.median(dt))


def _sanitize_time_and_q(t_in, q):
    if t_in.shape[0] != q.shape[0]:
        raise ValueError("time and q length mismatch")
    order = np.argsort(t_in)
    t_sorted = t_in[order]
    q_sorted = q[order]

    # Keep the first sample for each timestamp to avoid zero dt issues.
    unique_mask = np.concatenate([[True], np.diff(t_sorted) > 1e-9])
    t_unique = t_sorted[unique_mask]
    q_unique = q_sorted[unique_mask]

    if t_unique.shape[0] < 2:
        raise ValueError("not enough valid time samples after sanitization")
    return t_unique.astype(np.float32), q_unique.astype(np.float32)


def _lowpass_filter(q, fps, cutoff_hz):
    if cutoff_hz is None or cutoff_hz <= 0:
        return q
    try:
        from scipy.signal import butter, filtfilt
    except Exception:
        warnings.warn("scipy not available; skipping lowpass")
        return q

    nyq = 0.5 * fps
    if cutoff_hz >= nyq:
        warnings.warn(f"lowpass_hz={cutoff_hz} >= nyquist={nyq:.3f}; skipping lowpass")
        return q

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
    if clip_len <= 0 or stride <= 0:
        raise ValueError("clip_len and stride must be positive")
    clips = []
    for start in range(0, q.shape[0] - clip_len + 1, stride):
        clips.append((start, q[start : start + clip_len]))
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
    context_num_keys = list(manifest.get("context_num_keys", []))

    clip_len = int(round(clip_sec * fps_out))
    stride = int(round(stride_sec * fps_out))

    items = manifest.get("items", [])
    context_key_dims = _infer_context_key_dims(items, context_num_keys)
    context_dim = int(sum(context_key_dims.values()))

    rows = []
    all_q = []
    all_delta = []

    for item in items:
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
        unit = item.get("angle_unit", angle_unit)
        if unit == "deg":
            q = np.deg2rad(q).astype(np.float32)

        t_in, q = _sanitize_time_and_q(t_in, q)
        source_fps = _estimate_fps(t_in)

        if lowpass_hz:
            q = _lowpass_filter(q, fps=source_fps, cutoff_hz=lowpass_hz)

        q = _resample(q, t_in, fps_out)

        clips = _slice_clips(q, clip_len, stride)
        if not clips:
            warnings.warn(f"no clips produced for {demo_id}")
            continue

        context = item.get("context", {})
        context_numeric = _build_context_vector(context, context_num_keys, context_key_dims)

        for clip_idx, (start_idx, q_clip) in enumerate(clips):
            q_clip = q_clip.astype(np.float32)
            delta_q = np.zeros_like(q_clip)
            delta_q[1:] = q_clip[1:] - q_clip[:-1]
            dq = np.zeros_like(q_clip)
            dq[1:] = (q_clip[1:] - q_clip[:-1]) * fps_out
            q_goal = q_clip[-1]
            dq0 = dq[0]

            rows.append(
                {
                    "clip_id": f"{demo_id}_clip_{clip_idx:04d}",
                    "demo_id": demo_id,
                    "fps": fps_out,
                    "T": clip_len,
                    "joint_names": list(joint_names),
                    "q": q_clip.tolist(),
                    "delta_q": delta_q.tolist(),
                    "dq": dq.tolist(),
                    "q0": q_clip[0].tolist(),
                    "dq0": dq0.tolist(),
                    "q_goal": q_goal.tolist(),
                    "clip_start_idx": int(start_idx),
                    "clip_end_idx": int(start_idx + clip_len - 1),
                    "clip_start_sec": float(start_idx / fps_out),
                    "emotion_valence": float(item.get("emotion", {}).get("valence", 0.0)),
                    "emotion_arousal": float(item.get("emotion", {}).get("arousal", 0.0)),
                    "emotion_text": item.get("emotion", {}).get("text", ""),
                    "task_type": context.get("task_type", "idle"),
                    "context_numeric": context_numeric,
                    "context_json": json.dumps(context, ensure_ascii=True),
                    "meta_json": json.dumps(
                        {
                            "source_fps": source_fps,
                            "angle_unit": unit,
                            "time_col": time_col,
                            "joint_cols": list(joint_names),
                            "use_delta_q": use_delta_q,
                            "context_num_keys": context_num_keys,
                            "context_key_dims": context_key_dims,
                            "context_dim": context_dim,
                        },
                        ensure_ascii=True,
                    ),
                }
            )

            all_q.append(q_clip)
            all_delta.append(delta_q)

    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_parquet(out_parquet, index=False)

    if not all_q:
        raise RuntimeError("no clips found; stats not computed")

    q_cat = np.concatenate(all_q, axis=0)
    delta_cat = np.concatenate(all_delta, axis=0)

    stats = {
        "q_mean": q_cat.mean(axis=0).astype(np.float32).tolist(),
        "q_std": (q_cat.std(axis=0) + 1e-6).astype(np.float32).tolist(),
        "delta_mean": delta_cat.mean(axis=0).astype(np.float32).tolist(),
        "delta_std": (delta_cat.std(axis=0) + 1e-6).astype(np.float32).tolist(),
        "fps": fps_out,
        "context_dim": context_dim,
        "context_num_keys": context_num_keys,
        "context_key_dims": context_key_dims,
    }

    out_stats = Path(args.out_stats)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, indent=2))

    print(f"wrote {out_parquet}")
    print(f"wrote {out_stats}")


if __name__ == "__main__":
    main()
