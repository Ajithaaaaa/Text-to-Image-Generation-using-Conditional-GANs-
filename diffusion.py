# diffusion.py
import torch
import torch.nn.functional as F
import math
from configs import TIMESTEPS, BETA_START, BETA_END
import numpy as np

def linear_beta_schedule(timesteps=1000, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

betas = linear_beta_schedule(TIMESTEPS, BETA_START, BETA_END)
alphas = 1.0 - betas
alpha_prod = torch.cumprod(alphas, dim=0)
alpha_prod_prev = F.pad(alpha_prod[:-1], (1,0), value=1.0)

sqrt_alpha_prod = torch.sqrt(alpha_prod)
sqrt_one_minus_alpha_prod = torch.sqrt(1.0 - alpha_prod)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu()).to(t.device)
    return out.reshape(batch_size, *((1,)*(len(x_shape)-1)))

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alpha = get_index_from_list(sqrt_alpha_prod, t, x_start.shape)
    sqrt_one_minus = get_index_from_list(sqrt_one_minus_alpha_prod, t, x_start.shape)
    return sqrt_alpha * x_start + sqrt_one_minus * noise

def p_losses(denoise_model, x_start, t, noise=None, text_emb=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted = denoise_model(x_noisy, t, text_emb)
    # model predicts noise
    loss = F.mse_loss(predicted, noise)
    return loss
