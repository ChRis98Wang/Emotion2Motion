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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ckpt = torch.load(cfg["ckpt_path"], map_location="cpu")

    cond_cfg = cfg.get("cond", {})
    emotion = cond_cfg.get("emotion", {})
    task_type = cond_cfg.get("task_type", "idle")
    q0 = np.array(cond_cfg.get("q0", []), dtype=np.float32)

    task_vocab = ckpt.get("task_vocab", {})
    if task_type not in task_vocab:
        # fallback to first task
        task_id = 0
    else:
        task_id = int(task_vocab[task_type])

    q_dim = int(ckpt.get("q_dim", 4))
    if q0.size == 0:
        q0 = np.zeros((q_dim,), dtype=np.float32)
    if q0.shape[0] != q_dim:
        q0 = np.pad(q0, (0, max(0, q_dim - q0.shape[0])), mode="constant")[:q_dim]

    cond_dim = int(ckpt.get("cond_dim", 128))
    cond_encoder = ConditionEncoder(num_tasks=len(task_vocab), q_dim=q_dim, cond_dim=cond_dim)
    backbone = UNet1D(
        input_channels=q_dim,
        cond_dim=cond_dim,
        base_channels=int(ckpt.get("config", {}).get("model", {}).get("base_channels", 64)),
        num_down=int(ckpt.get("config", {}).get("model", {}).get("num_down", 2)),
        num_res_blocks=int(ckpt.get("config", {}).get("model", {}).get("num_res_blocks", 2)),
    )
    cond_encoder.load_state_dict(ckpt["cond_state"])
    backbone.load_state_dict(ckpt["model_state"])

    flow = RectifiedFlow(backbone, cond_encoder)
    flow.eval()

    T = int(round(float(cfg.get("clip_sec", 4.0)) * float(cfg.get("fps", 60))))
    steps = int(cfg.get("sampling_steps", 4))

    emotion_vec = torch.tensor([[float(emotion.get("valence", 0.0)), float(emotion.get("arousal", 0.0))]], dtype=torch.float32)
    task_id_t = torch.tensor([task_id], dtype=torch.long)
    q0_t = torch.tensor(q0[None, :], dtype=torch.float32)

    with torch.no_grad():
        x = flow.sample((1, T, q_dim), emotion_vec, task_id_t, q0_t, steps=steps)
    x_np = x.squeeze(0).cpu().numpy()

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
        "T": T,
        "fps": int(cfg.get("fps", 60)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
