import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        num_tasks,
        q_dim,
        cond_dim=128,
        task_embed_dim=16,
        hidden_dim=256,
        context_dim=0,
        use_q0=True,
        use_dq0=True,
        use_q_goal=True,
    ):
        super().__init__()
        self.num_tasks = int(max(1, num_tasks))
        self.q_dim = int(q_dim)
        self.cond_dim = int(cond_dim)
        self.context_dim = int(max(0, context_dim))
        self.use_q0 = bool(use_q0)
        self.use_dq0 = bool(use_dq0)
        self.use_q_goal = bool(use_q_goal)

        self.task_embed = nn.Embedding(self.num_tasks, int(task_embed_dim))

        in_dim = 2 + int(task_embed_dim)
        if self.use_q0:
            in_dim += self.q_dim
        if self.use_dq0:
            in_dim += self.q_dim
        if self.use_q_goal:
            in_dim += self.q_dim
        in_dim += self.context_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.cond_dim),
        )

    def _zero_like(self, ref, dim):
        return torch.zeros((ref.shape[0], dim), dtype=ref.dtype, device=ref.device)

    def forward(self, emotion, task_id, q0=None, dq0=None, q_goal=None, context_numeric=None):
        # emotion: [B,2], task_id: [B]
        if emotion.dim() != 2 or emotion.shape[-1] != 2:
            raise ValueError(f"emotion must be [B,2], got {tuple(emotion.shape)}")

        te = self.task_embed(task_id)
        feats = [emotion, te]

        if self.use_q0:
            if q0 is None:
                q0 = self._zero_like(emotion, self.q_dim)
            feats.append(q0)

        if self.use_dq0:
            if dq0 is None:
                dq0 = self._zero_like(emotion, self.q_dim)
            feats.append(dq0)

        if self.use_q_goal:
            if q_goal is None:
                q_goal = q0 if q0 is not None else self._zero_like(emotion, self.q_dim)
            feats.append(q_goal)

        if self.context_dim > 0:
            if context_numeric is None:
                context_numeric = self._zero_like(emotion, self.context_dim)
            elif context_numeric.shape[-1] != self.context_dim:
                if context_numeric.shape[-1] < self.context_dim:
                    pad = self.context_dim - context_numeric.shape[-1]
                    context_numeric = torch.nn.functional.pad(context_numeric, (0, pad))
                else:
                    context_numeric = context_numeric[:, : self.context_dim]
            feats.append(context_numeric)

        x = torch.cat(feats, dim=-1)
        return self.mlp(x)
