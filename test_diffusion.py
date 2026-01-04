print("🚀 Starting diffusion test...")

from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion (this will download the model first time ~2-4GB)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

pipe.to("cpu")

prompt = "A fantasy castle on a hill, colorful and detailed"
image = pipe(prompt).images[0]

image.save("test_output.png")
print("✅ Image saved as test_output.png")
