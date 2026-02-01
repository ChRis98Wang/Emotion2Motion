#!/usr/bin/env python3
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emg.models.cond import ConditionEncoder
from emg.models.unet1d import UNet1D
from emg.models.flow_matching import RectifiedFlow


def main():
    B, T, J = 2, 240, 4
    num_tasks = 3
    cond_dim = 128

    cond_encoder = ConditionEncoder(num_tasks=num_tasks, q_dim=J, cond_dim=cond_dim)
    backbone = UNet1D(input_channels=J, cond_dim=cond_dim)
    flow = RectifiedFlow(backbone, cond_encoder)

    x0 = torch.randn(B, T, J)
    emotion = torch.randn(B, 2)
    task_id = torch.tensor([0, 1], dtype=torch.long)
    q0 = torch.randn(B, J)

    loss, v_pred = flow.compute_loss(x0, emotion, task_id, q0)
    if v_pred.shape != (B, T, J):
        print(f"FAIL v_pred shape {v_pred.shape}")
        sys.exit(1)
    if not torch.isfinite(loss):
        print("FAIL loss not finite")
        sys.exit(1)

    print("PASS stage2")


if __name__ == "__main__":
    main()
