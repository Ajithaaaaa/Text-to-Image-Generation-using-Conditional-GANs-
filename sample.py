# sample.py
import torch
import torch.nn.functional as F
from configs import *
from model import UNet
from text_encoder import CLIPTextEncoder
from diffusion import betas, alphas, alpha_prod, sqrt_alpha_prod, sqrt_one_minus_alpha_prod, TIMESTEPS, get_index_from_list
from utils import denormalize, save_batch_images
import os
from tqdm import trange

def load_model(path, device=DEVICE):
    model = UNet(dim=64, channels=CHANNELS, dim_mults=(1,2,4), time_emb_dim=128, text_emb_dim=512).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model

def p_sample(model, x, t, text_emb, guidance_scale=1.0):
    """
    Predict x_{t-1} from x_t
    For classifier-free guidance we will call model twice: with text and with empty conditioning.
    """
    device = x.device
    t_tensor = torch.full((x.shape[0],), t, dtype=torch.long, device=device)
    # guidance: get conditioned and unconditioned predictions
    if guidance_scale != 1.0:
        # prepare text and empty embeddings
        # text_emb: (B, dim) -- we need an 'empty' embedding
        # Create empty embedding zeros of same dim
        empty_emb = torch.zeros_like(text_emb)
        emb = torch.cat([text_emb, empty_emb], dim=0)
        x_cat = torch.cat([x, x], dim=0)
        preds = model(x_cat, t_tensor.repeat(2), emb)
        pred_text, pred_uncond = preds.chunk(2, dim=0)
        pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)
    else:
        pred = model(x, t_tensor, text_emb)
    # Use predicted noise to compute mean of p(x_{t-1} | x_t)
    beta_t = betas[t].to(device)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_prod[t].to(device)
    alpha_cumprod_t = alpha_prod[t].to(device)
    alpha_cumprod_prev_t = alpha_prod[t-1] if t > 0 else torch.tensor(1.0).to(device)
    # Following simple DDPM posterior mean formula
    coef1 = (1 / torch.sqrt(alphas[t])) * ( (1 - alpha_prod[t-1]) / (1 - alpha_prod[t]) )
    # For simple implementation use q_sample inversion:
    # x0_pred = (x_t - sqrt(1 - alpha_prod_t) * pred) / sqrt(alpha_prod_t)
    x0_pred = (x - sqrt_one_minus_alpha_cumprod_t * pred) / torch.sqrt(alpha_cumprod_t)
    # clip
    x0_pred = torch.clamp(x0_pred, -1., 1.)
    if t == 0:
        return x0_pred
    posterior_variance = betas[t] * (1. - alpha_prod[t-1]) / (1. - alpha_prod[t])
    mean_coef1 = (torch.sqrt(alpha_prod_prev_t) * betas[t]) / (1. - alpha_prod[t])
    mean_coef2 = (torch.sqrt(alphas[t]) * (1. - alpha_prod_prev_t)) / (1. - alpha_prod[t])
    mean = mean_coef1 * x0_pred + mean_coef2 * x / torch.sqrt(alphas[t])
    noise = torch.randn_like(x)
    return mean + torch.sqrt(posterior_variance) * noise

@torch.no_grad()
def sample(model, text_prompts, text_encoder, batch_size=8, guidance_scale=GUIDANCE_SCALE, device=DEVICE):
    model.eval()
    text_emb = text_encoder.encode(text_prompts)  # (B, dim)
    x = torch.randn(batch_size, CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    for t in trange(TIMESTEPS-1, -1, -1):
        x = p_sample(model, x, t, text_emb, guidance_scale=guidance_scale)
    return x

if __name__ == "__main__":
    # example usage:
    device = DEVICE
    ckpt = "checkpoints/ckpt_final.pt"  # change to your checkpoint
    model = load_model(ckpt, device)
    text_encoder = CLIPTextEncoder(device=device)
    prompts = ["A red bicycle on a sunny street", "A futuristic city skyline at night"]
    imgs = sample(model, prompts, text_encoder, batch_size=len(prompts), guidance_scale=5.0, device=device)
    os.makedirs("samples", exist_ok=True)
    save_batch_images(imgs, "samples/sample_grid.png", nrow=len(prompts))
    print("Saved samples/samples_grid.png")
