if __name__ == "__main__":

    print("Starting story-to-image generation...")

    from diffusers import StableDiffusionPipeline
    import torch
    import os

    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")  # if no GPU

    # Example story
    story = """
    In a quiet village, a little boy found an ancient map hidden inside a dusty book.
    The map showed a secret path through the forest leading to a glowing waterfall.
    As he followed the path, fireflies lit the night sky, guiding him forward.
    Finally, he reached the waterfall, where magical creatures danced under the moonlight.
    """

    # Split into sentences
    scenes = [s.strip() for s in story.split("\n") if s.strip()]

    # Make output folder
    os.makedirs("story_images", exist_ok=True)

    # Generate images
    for i, scene in enumerate(scenes, start=1):
        print(f"Generating scene {i}: {scene}")
        image = pipe(scene).images[0]
        image.save(f"story_images/scene_{i}.png")

    print("Story images saved in 'story_images' folder")
