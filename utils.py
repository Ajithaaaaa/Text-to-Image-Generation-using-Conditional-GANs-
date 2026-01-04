# utils.py
import torch
import os
from torchvision.utils import save_image
from PIL import Image
import numpy as np

def save_checkpoint(model, optimizer, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'opt_state': optimizer.state_dict(),
        'step': step
    }, path)

def load_checkpoint(path, model, optimizer=None, device='cuda'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['opt_state'])
    return ckpt.get('step', 0)

def denormalize(x):
    # x in [-1,1] => [0,1]
    return (x.clamp(-1,1) + 1) / 2

def save_batch_images(x, filename, nrow=8):
    x = denormalize(x)
    save_image(x, filename, nrow=nrow)
