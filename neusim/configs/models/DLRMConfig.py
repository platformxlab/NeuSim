from pydantic import BaseModel
from pydantic import TypeAdapter

from neusim.configs.models.ModelConfig import ModelConfig


class MLPLayerConfig(BaseModel):
    in_features: int
    out_features: int
    bias: bool = True
    activation: str = "relu"


class DLRMConfig(ModelConfig):
    embedding_dim: int
    num_indices_per_lookup: list[int]
    embedding_table_sizes: list[int]

    num_dense_features: int
    bottom_mlp_config: list[MLPLayerConfig]
    top_mlp_config: list[MLPLayerConfig]

    interaction: str = "dot"

    model_type: str = "dlrm"

    use_vu_for_small_matmul: bool = False
