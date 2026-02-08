#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emg.datasets.clip_dataset import ClipDataset
from emg.datasets.collate import collate_batch
from emg.models.cond import ConditionEncoder
from emg.models.unet1d import UNet1D
from emg.models.flow_matching import RectifiedFlow
from emg.infer.postprocess import postprocess_clip


def load_limits(infer_cfg_path=None):
    defaults = {
        "q_min": [-1.57, -1.57, -1.57, -1.57],
        "q_max": [1.57, 1.57, 1.57, 1.57],
        "dq_max": [2.0, 2.0, 2.0, 2.0],
        "smooth_window": 0,
    }
    if infer_cfg_path and Path(infer_cfg_path).exists():
        cfg = yaml.safe_load(Path(infer_cfg_path).read_text())
        out = cfg.get("postprocess", {})
        for key, val in defaults.items():
            out.setdefault(key, val)
        return out
    return defaults


def compute_metrics(q, fps, q_min, q_max, dq_max):
    q = np.asarray(q, dtype=np.float32)
    q_min = np.asarray(q_min, dtype=np.float32)
    q_max = np.asarray(q_max, dtype=np.float32)
    dq_max = np.asarray(dq_max, dtype=np.float32)

    violations = (q < q_min) | (q > q_max)
    violation_rate = float(violations.mean())

    dq = np.diff(q, axis=0) * fps
    max_dq = float(np.max(np.abs(dq))) if dq.size else 0.0
    ddq = np.diff(dq, axis=0)
    smoothness = float(np.mean(np.linalg.norm(ddq, axis=-1))) if ddq.size else 0.0
    vel_violation = (np.abs(dq) > dq_max).mean() if dq.size else 0.0

    return {
        "limit_violation_rate": violation_rate,
        "vel_violation_rate": float(vel_violation),
        "max_abs_dq": max_dq,
        "smoothness": smoothness,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--infer_config", default=None)
    ap.add_argument("--report", default="data/tmp/report.json")
    ap.add_argument("--num_cond", type=int, default=8)
    ap.add_argument("--samples_per_cond", type=int, default=3)
    ap.add_argument("--sampling_steps", type=int, default=4)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})
    cond_meta = ckpt.get("condition_meta", {})

    dataset = ClipDataset(
        args.parquet,
        stats_path=args.stats,
        use_delta_q=cfg.get("dataset", {}).get("use_delta_q", True),
    )
    num_tasks = len(dataset.task_vocab)
    q_dim = dataset[0]["x"].shape[-1]

    cond_dim = int(ckpt.get("cond_dim", 128))
    cond_encoder = ConditionEncoder(
        num_tasks=max(1, num_tasks),
        q_dim=q_dim,
        cond_dim=cond_dim,
        hidden_dim=int(cfg.get("model", {}).get("cond_hidden_dim", 256)),
        context_dim=int(cond_meta.get("context_dim", dataset.context_dim)),
        use_q0=bool(cond_meta.get("use_q0", True)),
        use_dq0=bool(cond_meta.get("use_dq0", True)),
        use_q_goal=bool(cond_meta.get("use_q_goal", True)),
    )
    backbone = UNet1D(
        input_channels=q_dim,
        cond_dim=cond_dim,
        base_channels=int(cfg.get("model", {}).get("base_channels", 64)),
        num_down=int(cfg.get("model", {}).get("num_down", 2)),
        num_res_blocks=int(cfg.get("model", {}).get("num_res_blocks", 2)),
    )

    cond_encoder.load_state_dict(ckpt["cond_state"])
    backbone.load_state_dict(ckpt["model_state"])
    flow = RectifiedFlow(backbone, cond_encoder)
    flow.eval()

    limits = load_limits(args.infer_config)
    fps = int(dataset.stats.get("fps", 60)) if dataset.stats else 60

    num_cond = min(args.num_cond, len(dataset))
    indices = list(range(num_cond))

    all_metrics = []
    diversity_vals = []
    sensitivity_vals = []

    for idx in indices:
        item = dataset[idx]
        batch = collate_batch([item], dataset.task_vocab, device=None)

        base_kwargs = {
            "q0": batch["q0"],
            "dq0": batch["dq0"],
            "q_goal": batch["q_goal"],
            "context_numeric": batch["context_numeric"],
        }

        samples = []
        for _ in range(args.samples_per_cond):
            x = flow.sample(
                (1, batch["x"].shape[1], q_dim),
                batch["emotion"],
                batch["task_id"],
                steps=args.sampling_steps,
                **base_kwargs,
            )
            x_np = x.squeeze(0).detach().cpu().numpy()
            q = postprocess_clip(
                x_np,
                batch["q0"].squeeze(0).numpy(),
                stats_path=args.stats,
                use_delta_q=cfg.get("dataset", {}).get("use_delta_q", True),
                fps=fps,
                limits=limits,
            )
            samples.append(q)
            all_metrics.append(compute_metrics(q, fps, limits["q_min"], limits["q_max"], limits["dq_max"]))

        if len(samples) >= 2:
            dists = []
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    d = np.mean(np.linalg.norm(samples[i] - samples[j], axis=-1))
                    dists.append(d)
            diversity_vals.append(float(np.mean(dists)))

        # Sensitivity computed on generated trajectories for controlled condition changes.
        emo_lo = batch["emotion"].clone()
        emo_hi = batch["emotion"].clone()
        emo_lo[:, 0] = torch.clamp(batch["emotion"][:, 0] - 0.4, -1.0, 1.0)
        emo_hi[:, 0] = torch.clamp(batch["emotion"][:, 0] + 0.4, -1.0, 1.0)

        x_lo = flow.sample(
            (1, batch["x"].shape[1], q_dim),
            emo_lo,
            batch["task_id"],
            steps=args.sampling_steps,
            **base_kwargs,
        )
        x_hi = flow.sample(
            (1, batch["x"].shape[1], q_dim),
            emo_hi,
            batch["task_id"],
            steps=args.sampling_steps,
            **base_kwargs,
        )

        q_lo = postprocess_clip(
            x_lo.squeeze(0).detach().cpu().numpy(),
            batch["q0"].squeeze(0).numpy(),
            stats_path=args.stats,
            use_delta_q=cfg.get("dataset", {}).get("use_delta_q", True),
            fps=fps,
            limits=limits,
        )
        q_hi = postprocess_clip(
            x_hi.squeeze(0).detach().cpu().numpy(),
            batch["q0"].squeeze(0).numpy(),
            stats_path=args.stats,
            use_delta_q=cfg.get("dataset", {}).get("use_delta_q", True),
            fps=fps,
            limits=limits,
        )
        sensitivity_vals.append(float(np.mean(np.linalg.norm(q_lo - q_hi, axis=-1))))

    report = {
        "limit_violation_rate": float(np.mean([m["limit_violation_rate"] for m in all_metrics])) if all_metrics else 0.0,
        "max_abs_dq": float(np.max([m["max_abs_dq"] for m in all_metrics])) if all_metrics else 0.0,
        "smoothness": float(np.mean([m["smoothness"] for m in all_metrics])) if all_metrics else 0.0,
        "condition_diversity": float(np.mean(diversity_vals)) if diversity_vals else 0.0,
        "condition_sensitivity": float(np.mean(sensitivity_vals)) if sensitivity_vals else 0.0,
    }

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
