import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _load_stats(stats_path):
    if stats_path is None:
        return None
    stats = json.loads(Path(stats_path).read_text())
    return {
        "q_mean": np.array(stats.get("q_mean", []), dtype=np.float32),
        "q_std": np.array(stats.get("q_std", []), dtype=np.float32),
        "delta_mean": np.array(stats.get("delta_mean", []), dtype=np.float32),
        "delta_std": np.array(stats.get("delta_std", []), dtype=np.float32),
        "fps": int(stats.get("fps", 60)),
        "context_dim": int(stats.get("context_dim", 0)),
        "context_num_keys": list(stats.get("context_num_keys", [])),
    }


def build_task_vocab(task_types):
    uniq = sorted(set(task_types))
    return {name: idx for idx, name in enumerate(uniq)}


def _to_array(x):
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = np.stack(arr, axis=0)
    return arr.astype(np.float32)


def _ensure_1d(x, length):
    arr = _to_array(x).reshape(-1)
    if length == 0:
        return np.zeros((0,), dtype=np.float32)
    if arr.size < length:
        arr = np.pad(arr, (0, length - arr.size), mode="constant")
    elif arr.size > length:
        arr = arr[:length]
    return arr.astype(np.float32)


class ClipDataset(Dataset):
    def __init__(self, parquet_path, stats_path=None, use_delta_q=True):
        self.parquet_path = parquet_path
        self.df = pd.read_parquet(parquet_path)
        self.use_delta_q = use_delta_q
        self.stats = _load_stats(stats_path)

        self.task_types = self.df["task_type"].tolist()
        self.task_vocab = build_task_vocab(self.task_types)

        self.q_dim = int(_to_array(self.df.iloc[0]["q0"]).reshape(-1).shape[0])
        if "context_numeric" in self.df.columns:
            self.context_dim = int(_to_array(self.df.iloc[0]["context_numeric"]).reshape(-1).shape[0])
        elif self.stats is not None:
            self.context_dim = int(self.stats.get("context_dim", 0))
        else:
            self.context_dim = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q = _to_array(row["q"])
        delta_q = _to_array(row["delta_q"])
        q0 = _ensure_1d(row["q0"], self.q_dim)

        if self.use_delta_q:
            x = delta_q
            if self.stats is not None and self.stats["delta_std"].size:
                x = (x - self.stats["delta_mean"]) / self.stats["delta_std"]
        else:
            x = q
            if self.stats is not None and self.stats["q_std"].size:
                x = (x - self.stats["q_mean"]) / self.stats["q_std"]

        if "dq0" in row:
            dq0 = _ensure_1d(row["dq0"], self.q_dim)
        else:
            dq = np.zeros_like(q, dtype=np.float32)
            dq[1:] = (q[1:] - q[:-1]) * int(row["fps"])
            dq0 = dq[0]

        if "q_goal" in row:
            q_goal = _ensure_1d(row["q_goal"], self.q_dim)
        else:
            q_goal = q[-1].astype(np.float32)

        if "context_numeric" in row:
            context_numeric = _ensure_1d(row["context_numeric"], self.context_dim)
        else:
            context_numeric = np.zeros((self.context_dim,), dtype=np.float32)

        sample = {
            "x": torch.from_numpy(x),
            "q": torch.from_numpy(q),
            "delta_q": torch.from_numpy(delta_q),
            "q0": torch.from_numpy(q0),
            "dq0": torch.from_numpy(dq0),
            "q_goal": torch.from_numpy(q_goal),
            "context_numeric": torch.from_numpy(context_numeric),
            "emotion": torch.tensor(
                [row["emotion_valence"], row["emotion_arousal"]], dtype=torch.float32
            ),
            "task_type": row["task_type"],
            "context_json": row["context_json"],
            "joint_names": row["joint_names"],
            "fps": int(row["fps"]),
            "clip_id": row.get("clip_id", ""),
            "demo_id": row.get("demo_id", ""),
        }
        return sample
