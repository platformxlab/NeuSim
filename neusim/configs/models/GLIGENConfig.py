from pydantic import BaseModel
from pydantic import TypeAdapter

from neusim.configs.models.ModelConfig import ModelConfig


class FourierEmbedderConfig(BaseModel):
    num_freqs: int = 64


class TextEmbedderConfig(BaseModel):
    d_model: int = 512
    num_heads: int = 8
    d_head: int = 64
    d_ff: int = 2048
    num_layers: int = 12
    ffn_type: str = "default"


class ImageEmbedderConfig(BaseModel):
    model_type: str = "vit"
    patch_size: int = 2
    d_model: int = 1024
    num_heads: int = 16
    d_head: int = 64
    d_ff: int = 4096
    num_layers: int = 24
    ffn_type: str = "default"


class SpatialConditionEmbedderConfig(BaseModel):
    class StemConfig(BaseModel):
        in_channels: int = 3
        out_channels: int = 96
        kernel_size: int = 4
        stride: int = 4

    model_type: str = "convnext"
    stem: StemConfig = StemConfig()
    depths: list[int] = [3, 3, 9, 3]
    dims: list[int] = [96, 192, 384, 768]


class GroundingInputConfig(BaseModel):
    class TextConfig(BaseModel):
        input_seqlen: int = 512
        feature_dim: int = 768

    class BboxConfig(BaseModel):
        input_seqlen: int = 8
        feature_dim: int = 4
        grounding_token_feature_dim: int = 768

    class ImageConfig(BaseModel):
        resolution: list[int] = [1024, 1024]
        image_num_channels: int = 3

    class KeypointConfig(BaseModel):
        num_persons: int = 10
        num_keypoints: int = 17
        feature_dim: int = 256

    class SpatialConditionConfig(BaseModel):
        resolution: list[int] = [256, 256]
        num_channels: int = 1

    text: TextConfig = TextConfig()
    bbox: BboxConfig = BboxConfig()
    image: ImageConfig = ImageConfig()
    keypoint: KeypointConfig = KeypointConfig()
    spatial_condition: SpatialConditionConfig = SpatialConditionConfig()


class UNetConfig(BaseModel):
    noisy_latent_resolution: list[int] = [64, 64]
    model_channels: int = 320
    attention_resolutions: list[int] = [4, 2, 1]
    num_res_blocks: int = 2
    channel_mult: list[int] = [1, 2, 4, 4]
    num_heads: int = 8
    context_dim: int = 768


class GLIGENConfig(ModelConfig):
    model_type: str = "gligen"

    num_diffusion_steps: int = 1
    total_num_diffusion_steps: int = 40
    image_resolution: list[int] = [512, 512]
    image_num_channels: int = 3
    use_flash_attention: bool = False

    fourier_embedder_config: FourierEmbedderConfig = FourierEmbedderConfig()
    text_embedder_config: TextEmbedderConfig = TextEmbedderConfig()
    image_embedder_config: ImageEmbedderConfig = ImageEmbedderConfig()
    spatial_condition_embedder_config: SpatialConditionEmbedderConfig = SpatialConditionEmbedderConfig()
    grounding_input_config: GroundingInputConfig = GroundingInputConfig()
    unet_config: UNetConfig = UNetConfig()

    output_dir: str = "./llava_ops"
