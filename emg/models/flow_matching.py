import torch
import torch.nn as nn


class RectifiedFlow(nn.Module):
    def __init__(self, backbone, cond_encoder):
        super().__init__()
        self.backbone = backbone
        self.cond_encoder = cond_encoder

    def forward(self, x, t, cond):
        return self.backbone(x, t, cond)

    def compute_loss(self, x0, emotion, task_id, q0=None, dq0=None, q_goal=None, context_numeric=None):
        # x0: [B,T,J]
        device = x0.device
        noise = torch.randn_like(x0)
        t = torch.rand((x0.shape[0], 1, 1), device=device)
        x_t = (1.0 - t) * noise + t * x0
        target_v = x0 - noise

        cond = self.cond_encoder(emotion, task_id, q0=q0, dq0=dq0, q_goal=q_goal, context_numeric=context_numeric)
        v_pred = self.backbone(x_t, t, cond)
        loss = torch.mean((v_pred - target_v) ** 2)
        return loss, v_pred

    @torch.no_grad()
    def sample(self, shape, emotion, task_id, q0=None, dq0=None, q_goal=None, context_numeric=None, steps=4):
        device = emotion.device
        x = torch.randn(shape, device=device)
        cond = self.cond_encoder(emotion, task_id, q0=q0, dq0=dq0, q_goal=q_goal, context_numeric=context_numeric)

        dt = 1.0 / float(steps)
        for i in range(steps):
            t = torch.full((shape[0], 1, 1), (i + 0.5) * dt, device=device)
            v = self.backbone(x, t, cond)
            x = x + v * dt
        return x
