import torch
import torch.nn as nn


class Transformer1D(nn.Module):
    def __init__(self, input_channels, cond_dim, model_dim=128, num_layers=4, num_heads=4):
        super().__init__()
        self.input_channels = input_channels
        self.cond_dim = cond_dim

        self.in_proj = nn.Linear(input_channels, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cond_proj = nn.Linear(cond_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, input_channels)

    def forward(self, x, t, cond):
        # x: [B,T,J]
        h = self.in_proj(x)
        cond_bias = self.cond_proj(cond)[:, None, :]
        h = h + cond_bias
        h = self.encoder(h)
        return self.out_proj(h)
