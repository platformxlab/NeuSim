from pydantic import TypeAdapter

from neusim.configs.models.LLMConfig import LLMConfig


class DiTConfig(LLMConfig):
    image_width: int
    num_channels: int
    patch_size: int
    num_diffusion_steps: int

    model_type: str = "dit"
    ffn_type: str = "default"

    input_seqlen: int = 0  # will be derived automatically
    output_seqlen: int = 0  # unused
