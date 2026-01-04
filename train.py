# train.py
import os
import torch
from torch import optim
from tqdm import tqdm
from configs import *
from dataset import get_dataloader
from text_encoder import CLIPTextEncoder
from model import UNet
from diffusion import p_losses
from utils import save_checkpoint
import random

def cycle_timesteps(batch_size, timesteps=TIMESTEPS, device='cuda'):
    return torch.randint(0, timesteps, (batch_size,), device=device, dtype=torch.long)

def train():
    device = DEVICE
    dataloader = get_dataloader("data/images", "data/captions.csv", img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    text_encoder = CLIPTextEncoder(device=device)
    model = UNet(dim=64, channels=CHANNELS, dim_mults=(1,2,4), time_emb_dim=128, text_emb_dim=512).to(device)
    opt = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)

    step = 0
    os.makedirs(OUT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, captions in pbar:
            model.train()
            imgs = imgs.to(device)
            bsz = imgs.shape[0]
            # text embeddings
            # classifier-free: with probability CFG_PROB, replace caption with empty string
            captions_cf = []
            for c in captions:
                if random.random() < CFG_PROB:
                    captions_cf.append("")  # drop conditioning
                else:
                    captions_cf.append(c)
            text_emb = text_encoder.encode(captions_cf)  # (B, dim)

            t = cycle_timesteps(bsz, TIMESTEPS, device)
            loss = p_losses(model, imgs, t, text_emb=text_emb)
            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            pbar.set_postfix({'loss': loss.item(), 'step': step})

            if step % SAVE_EVERY == 0:
                save_checkpoint(model, opt, step, os.path.join(OUT_DIR, f"ckpt_{step}.pt"))

            if step % SAMPLE_EVERY == 0:
                # save small sample using sampling script helper
                pass

    # final save
    save_checkpoint(model, opt, step, os.path.join(OUT_DIR, f"ckpt_final.pt"))

if __name__ == "__main__":
    train()
