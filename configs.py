# configs.py
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 64          # change to 128 if you have more compute; must match model
CHANNELS = 3
BATCH_SIZE = 32
LR = 2e-4
EPOCHS = 100
SAVE_EVERY = 1000      # steps
SAMPLE_EVERY = 1000    # steps
OUT_DIR = "checkpoints"

# diffusion hyperparams
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

# classifier-free guidance
CFG_PROB = 0.1  # probability to drop text conditioning during training
GUIDANCE_SCALE = 5.0  # at sampling
