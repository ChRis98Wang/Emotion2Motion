import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    if t.dim() == 2:
        t = t[:, 0]
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=t.device)], dim=-1)
    return emb


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        cond_out = self.cond_proj(cond)[:, :, None]
        h = h + cond_out
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet1D(nn.Module):
    def __init__(self, input_channels, cond_dim, base_channels=64, num_down=2, num_res_blocks=2, time_dim=32):
        super().__init__()
        self.input_channels = input_channels
        self.cond_dim = cond_dim
        self.time_dim = time_dim
        self.num_down = num_down
        self.num_res_blocks = num_res_blocks

        self.proj_in = nn.Conv1d(input_channels, base_channels, kernel_size=1)

        down_blocks = []
        downsamples = []
        ch = base_channels
        for _ in range(num_down):
            blocks = nn.ModuleList([ResBlock1D(ch, ch, cond_dim + time_dim) for _ in range(num_res_blocks)])
            down_blocks.append(blocks)
            downsamples.append(nn.Conv1d(ch, ch * 2, kernel_size=4, stride=2, padding=1))
            ch *= 2
        self.down_blocks = nn.ModuleList(down_blocks)
        self.downsamples = nn.ModuleList(downsamples)

        self.mid = ResBlock1D(ch, ch, cond_dim + time_dim)

        up_blocks = []
        upsamples = []
        for _ in range(num_down):
            upsamples.append(nn.ConvTranspose1d(ch, ch // 2, kernel_size=4, stride=2, padding=1))
            ch = ch // 2
            blocks = [ResBlock1D(ch * 2, ch, cond_dim + time_dim)]
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock1D(ch, ch, cond_dim + time_dim))
            up_blocks.append(nn.ModuleList(blocks))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.upsamples = nn.ModuleList(upsamples)

        self.proj_out = nn.Conv1d(base_channels, input_channels, kernel_size=1)

    def forward(self, x, t, cond):
        if x.dim() != 3:
            raise ValueError("x must be [B,T,J]")
        x = x.transpose(1, 2)

        if t.dim() == 3:
            t_in = t[:, 0, 0]
        elif t.dim() == 2:
            t_in = t[:, 0]
        else:
            t_in = t
        t_emb = timestep_embedding(t_in, self.time_dim)
        cond_full = torch.cat([cond, t_emb], dim=-1)

        h = self.proj_in(x)
        skips = []
        for level in range(self.num_down):
            for block in self.down_blocks[level]:
                h = block(h, cond_full)
            skips.append(h)
            h = self.downsamples[level](h)

        h = self.mid(h, cond_full)

        for level in range(self.num_down):
            h = self.upsamples[level](h)
            skip = skips.pop()
            if skip.shape[-1] != h.shape[-1]:
                diff = skip.shape[-1] - h.shape[-1]
                if diff > 0:
                    skip = skip[:, :, : h.shape[-1]]
                else:
                    h = F.pad(h, (0, -diff))
            h = torch.cat([h, skip], dim=1)
            for block in self.up_blocks[level]:
                h = block(h, cond_full)

        out = self.proj_out(h)
        return out.transpose(1, 2)
