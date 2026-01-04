# text_encoder.py
from transformers import CLIPTokenizer, CLIPTextModel
import torch

class CLIPTextEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts):
        # texts: list[str]
        tokenized = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(**tokenized)
        # use pooled output (last_hidden_state mean or pooled_output depending on model)
        # CLIPTextModel returns last_hidden_state and pooled_output: pooled_output is fine
        return outputs.pooler_output  # shape: (batch, dim)
