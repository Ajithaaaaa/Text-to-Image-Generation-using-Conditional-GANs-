# model.py
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def exist_or(first, second):
    return first if first is not None else second

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, text_emb_dim=None):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_ch),
            nn.SiLU()
        )
        self.text_mlp = None
        if text_emb_dim is not None:
            self.text_mlp = nn.Sequential(nn.Linear(text_emb_dim, out_ch), nn.SiLU())

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, t_emb, text_emb=None):
        h = self.block1(x)
        # add time embedding
        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t
        if self.text_mlp is not None and text_emb is not None:
            te = self.text_mlp(text_emb)[:, :, None, None]
            h = h + te
        h = self.block2(h)
        return h + self.res_conv(x)

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 4, 2, 1)  # halve spatial

    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, 2, 1)

    def forward(self, x):
        return self.op(x)

class UNet(nn.Module):
    def __init__(self, dim=64, channels=3, dim_mults=(1,2,4), time_emb_dim=128, text_emb_dim=512):
        super().__init__()
        self.channels = channels
        self.time_dim = time_emb_dim
        self.text_emb_dim = text_emb_dim

        self.input_conv = nn.Conv2d(channels, dim, 3, padding=1)
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU())

        # down
        dims = [dim * m for m in dim_mults]
        in_out = list(zip([dim] + dims[:-1], dims))
        self.downs = nn.ModuleList()
        for (i, (i_ch, o_ch)) in enumerate(in_out):
            self.downs.append(nn.ModuleList([
                ResidualBlock(i_ch, o_ch, time_emb_dim, text_emb_dim),
                ResidualBlock(o_ch, o_ch, time_emb_dim, text_emb_dim),
                Downsample(o_ch) if i < len(in_out)-1 else nn.Identity()
            ]))

        # bottleneck
        mid_ch = dims[-1]
        self.mid_block1 = ResidualBlock(mid_ch, mid_ch*2, time_emb_dim, text_emb_dim)
        self.mid_block2 = ResidualBlock(mid_ch*2, mid_ch, time_emb_dim, text_emb_dim)

        # up
        self.ups = nn.ModuleList()
        rev_in_out = list(zip(dims[::-1], [d for d in dims[::-1][1:]] + [dim]))
        for (i_ch, o_ch) in rev_in_out:
            self.ups.append(nn.ModuleList([
                ResidualBlock(i_ch*2, i_ch, time_emb_dim, text_emb_dim),
                ResidualBlock(i_ch, o_ch, time_emb_dim, text_emb_dim),
                Upsample(o_ch) if o_ch != dim else nn.Identity()
            ]))

        self.final_res = ResidualBlock(dim, dim, time_emb_dim, text_emb_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, x, t, text_emb=None):
        # x: (B,C,H,W), t: (B,), text_emb: (B, text_dim) or None
        t_emb = self.time_mlp(t)
        x = self.input_conv(x)
        hiddens = []
        # down
        for r1, r2, down in self.downs:
            x = r1(x, t_emb, text_emb)
            x = r2(x, t_emb, text_emb)
            hiddens.append(x)
            x = down(x)
        # mid
        x = self.mid_block1(x, t_emb, text_emb)
        x = self.mid_block2(x, t_emb, text_emb)
        # up
        for (r1, r2, up), hid in zip(self.ups, reversed(hiddens)):
            x = torch.cat([x, hid], dim=1)
            x = r1(x, t_emb, text_emb)
            x = r2(x, t_emb, text_emb)
            x = up(x)
        x = self.final_res(x, t_emb, text_emb)
        return self.final_conv(x)
