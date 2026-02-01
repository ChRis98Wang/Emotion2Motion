import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    def __init__(self, num_tasks, q_dim, cond_dim=128, task_embed_dim=16, hidden_dim=128):
        super().__init__()
        self.num_tasks = num_tasks
        self.q_dim = q_dim
        self.cond_dim = cond_dim

        self.task_embed = nn.Embedding(num_tasks, task_embed_dim)
        in_dim = 2 + task_embed_dim + q_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cond_dim),
        )

    def forward(self, emotion, task_id, q0):
        # emotion: [B,2], task_id: [B], q0: [B,J]
        te = self.task_embed(task_id)
        x = torch.cat([emotion, te, q0], dim=-1)
        return self.mlp(x)
