#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emg.models.cond import ConditionEncoder
from emg.models.unet1d import UNet1D
from emg.models.flow_matching import RectifiedFlow
from emg.infer.postprocess import postprocess_clip


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


def _build_context_vector(cond_cfg, context_keys, context_dim):
    if context_dim <= 0:
        return np.zeros((0,), dtype=np.float32)

    if "context_numeric" in cond_cfg:
        arr = np.asarray(cond_cfg.get("context_numeric", []), dtype=np.float32).reshape(-1)
    else:
        context = cond_cfg.get("context", {})
        vec = []
        for key in context_keys:
            val = _context_lookup(context, key)
            flat = _flatten_numeric(val)
            if not flat:
                flat = [0.0]
            vec.extend(flat)
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)

    if arr.size < context_dim:
        arr = np.pad(arr, (0, context_dim - arr.size), mode="constant")
    elif arr.size > context_dim:
        arr = arr[:context_dim]
    return arr.astype(np.float32)


def _resolve_device(device_opt):
    if device_opt == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but not available")
        return torch.device("cuda")
    if device_opt == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_dim(arr, dim, fill=0.0):
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size < dim:
        arr = np.pad(arr, (0, dim - arr.size), mode="constant", constant_values=fill)
    elif arr.size > dim:
        arr = arr[:dim]
    return arr.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--strict_task", action="store_true", help="raise on unknown task_type")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = _resolve_device(args.device)
    ckpt = torch.load(cfg["ckpt_path"], map_location=device)

    cond_cfg = cfg.get("cond", {})
    emotion = cond_cfg.get("emotion", {})
    task_type = cond_cfg.get("task_type", "idle")

    task_vocab = ckpt.get("task_vocab", {})
    if task_type not in task_vocab:
        if args.strict_task or bool(cfg.get("strict_task", False)):
            raise ValueError(f"task_type '{task_type}' not in task vocab: {sorted(task_vocab.keys())}")
        task_id = 0
    else:
        task_id = int(task_vocab[task_type])

    q_dim = int(ckpt.get("q_dim", 4))
    q0 = _ensure_dim(cond_cfg.get("q0", []), q_dim, fill=0.0)
    dq0 = _ensure_dim(cond_cfg.get("dq0", [0.0] * q_dim), q_dim, fill=0.0)
    q_goal = _ensure_dim(cond_cfg.get("q_goal", q0.tolist()), q_dim, fill=0.0)

    cond_meta = ckpt.get("condition_meta", {})
    context_dim = int(cond_meta.get("context_dim", 0))
    context_keys = list(cond_meta.get("context_num_keys", []))
    context_numeric = _build_context_vector(cond_cfg, context_keys, context_dim)

    cond_dim = int(ckpt.get("cond_dim", 128))
    cond_encoder = ConditionEncoder(
        num_tasks=max(1, len(task_vocab)),
        q_dim=q_dim,
        cond_dim=cond_dim,
        hidden_dim=int(ckpt.get("config", {}).get("model", {}).get("cond_hidden_dim", 256)),
        context_dim=context_dim if bool(cond_meta.get("use_context", True)) else 0,
        use_q0=bool(cond_meta.get("use_q0", True)),
        use_dq0=bool(cond_meta.get("use_dq0", True)),
        use_q_goal=bool(cond_meta.get("use_q_goal", True)),
    )
    backbone = UNet1D(
        input_channels=q_dim,
        cond_dim=cond_dim,
        base_channels=int(ckpt.get("config", {}).get("model", {}).get("base_channels", 64)),
        num_down=int(ckpt.get("config", {}).get("model", {}).get("num_down", 2)),
        num_res_blocks=int(ckpt.get("config", {}).get("model", {}).get("num_res_blocks", 2)),
    )
    cond_encoder.load_state_dict(ckpt["cond_state"])
    backbone.load_state_dict(ckpt["model_state"])

    flow = RectifiedFlow(backbone, cond_encoder).to(device)
    flow.eval()

    T = int(round(float(cfg.get("clip_sec", 4.0)) * float(cfg.get("fps", 60))))
    steps = int(cfg.get("sampling_steps", 4))

    emotion_vec = torch.tensor(
        [[float(emotion.get("valence", 0.0)), float(emotion.get("arousal", 0.0))]],
        dtype=torch.float32,
        device=device,
    )
    task_id_t = torch.tensor([task_id], dtype=torch.long, device=device)
    q0_t = torch.tensor(q0[None, :], dtype=torch.float32, device=device)
    dq0_t = torch.tensor(dq0[None, :], dtype=torch.float32, device=device)
    q_goal_t = torch.tensor(q_goal[None, :], dtype=torch.float32, device=device)
    context_t = torch.tensor(context_numeric[None, :], dtype=torch.float32, device=device)

    with torch.no_grad():
        x = flow.sample(
            (1, T, q_dim),
            emotion_vec,
            task_id_t,
            q0=q0_t,
            dq0=dq0_t,
            q_goal=q_goal_t,
            context_numeric=context_t,
            steps=steps,
        )
    x_np = x.squeeze(0).detach().cpu().numpy()

    q = postprocess_clip(
        x_np,
        q0,
        stats_path=cfg.get("stats_path"),
        use_delta_q=ckpt.get("config", {}).get("dataset", {}).get("use_delta_q", True),
        fps=int(cfg.get("fps", 60)),
        limits=cfg.get("postprocess", {}),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "q_clip.csv"
    df = pd.DataFrame(q, columns=[f"J{i+1}" for i in range(q_dim)])
    df.to_csv(out_csv, index=False)

    meta = {
        "task_type": task_type,
        "emotion": emotion,
        "q0": q0.tolist(),
        "dq0": dq0.tolist(),
        "q_goal": q_goal.tolist(),
        "context_numeric": context_numeric.tolist(),
        "T": T,
        "fps": int(cfg.get("fps", 60)),
        "device": str(device),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
