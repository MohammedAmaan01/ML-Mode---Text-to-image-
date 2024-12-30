from diffusers import StableDiffusionPipeline
import torch

def load_stable_diffusion_pipeline(model_id: str, auth_token: str, device: str = "cuda"):
    """
    Load the Stable Diffusion pipeline with the given model ID and authentication token.

    Args:
        model_id (str): The ID of the Stable Diffusion model.
        auth_token (str): The Hugging Face authentication token.
        device (str): The device to load the model on ('cuda' or 'cpu').

    Returns:
        StableDiffusionPipeline: The loaded Stable Diffusion pipeline.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=auth_token
    )
    pipe.to(device)
    return pipe
