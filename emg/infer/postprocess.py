import json
from pathlib import Path
import numpy as np


def load_stats(stats_path):
    if stats_path is None:
        return None
    stats = json.loads(Path(stats_path).read_text())
    return {
        "q_mean": np.array(stats.get("q_mean", []), dtype=np.float32),
        "q_std": np.array(stats.get("q_std", []), dtype=np.float32),
        "delta_mean": np.array(stats.get("delta_mean", []), dtype=np.float32),
        "delta_std": np.array(stats.get("delta_std", []), dtype=np.float32),
        "fps": int(stats.get("fps", 60)),
    }


def denormalize(x, mean, std):
    return x * std + mean


def integrate_delta(delta_q, q0):
    q = np.zeros_like(delta_q, dtype=np.float32)
    q[0] = q0
    for t in range(1, delta_q.shape[0]):
        q[t] = q[t - 1] + delta_q[t]
    return q


def apply_joint_limits(q, q_min, q_max):
    q_min = np.asarray(q_min, dtype=np.float32)
    q_max = np.asarray(q_max, dtype=np.float32)
    return np.clip(q, q_min, q_max)


def apply_velocity_limits(q, fps, dq_max):
    dq_max = np.asarray(dq_max, dtype=np.float32)
    max_step = dq_max / float(fps)
    q_new = np.zeros_like(q, dtype=np.float32)
    q_new[0] = q[0]
    for t in range(1, q.shape[0]):
        delta = q[t] - q_new[t - 1]
        delta = np.clip(delta, -max_step, max_step)
        q_new[t] = q_new[t - 1] + delta
    return q_new


def smooth_trajectory(q, window):
    if window <= 1:
        return q
    kernel = np.ones(window, dtype=np.float32) / float(window)
    q_s = np.zeros_like(q, dtype=np.float32)
    for j in range(q.shape[1]):
        q_s[:, j] = np.convolve(q[:, j], kernel, mode="same")
    q_s[0] = q[0]
    return q_s


def postprocess_clip(x, q0, stats_path=None, use_delta_q=True, fps=60, limits=None):
    stats = load_stats(stats_path) if stats_path else None
    limits = limits or {}
    q_min = limits.get("q_min", [-3.14] * x.shape[1])
    q_max = limits.get("q_max", [3.14] * x.shape[1])
    dq_max = limits.get("dq_max", [5.0] * x.shape[1])
    smooth_window = int(limits.get("smooth_window", 0))

    if use_delta_q:
        delta = x
        if stats is not None and stats["delta_std"].size:
            delta = denormalize(delta, stats["delta_mean"], stats["delta_std"])
        q = integrate_delta(delta, q0)
    else:
        q = x
        if stats is not None and stats["q_std"].size:
            q = denormalize(q, stats["q_mean"], stats["q_std"])

    q = apply_joint_limits(q, q_min, q_max)
    q = apply_velocity_limits(q, fps, dq_max)
    q = apply_joint_limits(q, q_min, q_max)
    q = smooth_trajectory(q, smooth_window)
    q = apply_joint_limits(q, q_min, q_max)
    return q
