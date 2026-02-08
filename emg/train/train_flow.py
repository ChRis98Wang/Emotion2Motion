#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import random
import itertools
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from emg.datasets.clip_dataset import ClipDataset
from emg.datasets.collate import collate_batch
from emg.models.cond import ConditionEncoder
from emg.models.unet1d import UNet1D
from emg.models.flow_matching import RectifiedFlow


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--overfit", action="store_true", help="overfit a single batch for smoke tests")
    ap.add_argument("--tb_log_dir", default=None, help="tensorboard log dir (optional)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    use_cuda = bool(cfg.get("use_cuda", True)) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset_cfg = cfg.get("dataset", {})
    dataset = ClipDataset(
        dataset_cfg["parquet_path"],
        stats_path=dataset_cfg.get("stats_path"),
        use_delta_q=dataset_cfg.get("use_delta_q", True),
    )

    batch_size = args.batch_size or cfg.get("optim", {}).get("batch_size", 8)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: x
    )
    loader_iter = itertools.cycle(loader)

    num_tasks = len(dataset.task_vocab)
    sample0 = dataset[0]
    q_dim = sample0["x"].shape[-1]
    context_dim = int(dataset.context_dim)

    model_cfg = cfg.get("model", {})
    cond_cfg = cfg.get("condition", {})
    cond_dim = int(model_cfg.get("cond_dim", 128))
    cond_encoder = ConditionEncoder(
        num_tasks=num_tasks,
        q_dim=q_dim,
        cond_dim=cond_dim,
        hidden_dim=int(model_cfg.get("cond_hidden_dim", 256)),
        context_dim=context_dim if cond_cfg.get("use_context", True) else 0,
        use_q0=bool(cond_cfg.get("use_q0", True)),
        use_dq0=bool(cond_cfg.get("use_dq0", True)),
        use_q_goal=bool(cond_cfg.get("use_q_goal", True)),
    )
    backbone = UNet1D(
        input_channels=q_dim,
        cond_dim=cond_dim,
        base_channels=int(model_cfg.get("base_channels", 64)),
        num_down=int(model_cfg.get("num_down", 2)),
        num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
    )

    flow = RectifiedFlow(backbone, cond_encoder).to(device)

    optim_cfg = cfg.get("optim", {})
    lr = float(optim_cfg.get("lr", 1e-3))
    wd = float(optim_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=wd)

    train_cfg = cfg.get("training", {})
    num_steps = args.num_steps or int(train_cfg.get("num_steps", 400))
    log_interval = int(train_cfg.get("log_interval", 20))
    save_every = int(train_cfg.get("save_every", 200))
    save_dir = Path(args.save_dir or train_cfg.get("save_dir", "data/tmp/checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    tb_log_dir = args.tb_log_dir or train_cfg.get("tb_log_dir")
    writer = None
    if tb_log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as e:
            raise SystemExit(f"tensorboard not available: {e}")
        writer = SummaryWriter(log_dir=tb_log_dir)

    losses = []
    flow.train()
    fixed_batch = None
    if args.overfit:
        fixed_batch = collate_batch(next(loader_iter), dataset.task_vocab, device=device)

    for step in range(1, num_steps + 1):
        batch = fixed_batch if fixed_batch is not None else collate_batch(next(loader_iter), dataset.task_vocab, device=device)
        x = batch["x"]
        loss, _ = flow.compute_loss(
            x,
            batch["emotion"],
            batch["task_id"],
            q0=batch["q0"],
            dq0=batch["dq0"],
            q_goal=batch["q_goal"],
            context_numeric=batch["context_numeric"],
        )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(float(loss.item()))
        if step % log_interval == 0:
            print(f"step {step} loss {loss.item():.6f}")
        if writer is not None:
            writer.add_scalar("train/loss", float(loss.item()), step)
            writer.add_scalar("train/grad_norm", float(grad_norm), step)

        if step % save_every == 0:
            ckpt_path = save_dir / f"flow_step_{step}.pt"
            torch.save(
                {
                    "model_state": flow.backbone.state_dict(),
                    "cond_state": flow.cond_encoder.state_dict(),
                    "config": cfg,
                    "task_vocab": dataset.task_vocab,
                    "q_dim": q_dim,
                    "cond_dim": cond_dim,
                    "condition_meta": {
                        "context_dim": context_dim,
                        "use_q0": bool(cond_cfg.get("use_q0", True)),
                        "use_dq0": bool(cond_cfg.get("use_dq0", True)),
                        "use_q_goal": bool(cond_cfg.get("use_q_goal", True)),
                        "use_context": bool(cond_cfg.get("use_context", True)),
                        "context_num_keys": list(dataset.stats.get("context_num_keys", [])) if dataset.stats else [],
                    },
                },
                ckpt_path,
            )

    # save last
    last_path = save_dir / "flow_last.pt"
    torch.save(
        {
            "model_state": flow.backbone.state_dict(),
            "cond_state": flow.cond_encoder.state_dict(),
            "config": cfg,
            "task_vocab": dataset.task_vocab,
            "q_dim": q_dim,
            "cond_dim": cond_dim,
            "condition_meta": {
                "context_dim": context_dim,
                "use_q0": bool(cond_cfg.get("use_q0", True)),
                "use_dq0": bool(cond_cfg.get("use_dq0", True)),
                "use_q_goal": bool(cond_cfg.get("use_q_goal", True)),
                "use_context": bool(cond_cfg.get("use_context", True)),
                "context_num_keys": list(dataset.stats.get("context_num_keys", [])) if dataset.stats else [],
            },
            "losses": losses,
        },
        last_path,
    )

    log_path = save_dir / "train_log.json"
    log_path.write_text(json.dumps({"losses": losses}, indent=2))
    print(f"wrote {last_path}")
    print(f"wrote {log_path}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
