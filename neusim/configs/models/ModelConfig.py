import neusim.configs.chips.ChipConfig as ChipConfig
import neusim.configs.systems.SystemConfig as SystemConfig


# @dataclass(kw_only=True)
class ModelConfig(ChipConfig.ChipConfig, SystemConfig.SystemConfig):
    model_name: str = "model"
    model_type: str
    global_batch_size: int = 1

    num_chips: int = 1
    data_parallelism_degree: int = 1
    tensor_parallelism_degree: int = 1
    pipeline_parallelism_degree: int = 1
    num_data_parallel_axes: int = 1
    num_tensor_parallel_axes: int = 1
    num_pipeline_parallel_axes: int = 1
    data_parallel_degree_dcn: int = 1
    tensor_parallel_degree_dcn: int = 1
    pipeline_parallel_degree_dcn: int = 1
    microbatch_size_dcn: int = 1
    microbatch_size_ici: int = 1

    output_file_path: str = "./output.csv"

    def __hash__(self) -> int:
        '''
        Just hash some critical fields of the config.
        This is probably not the most efficient way, so use it with caution.
        '''
        return hash(
            (
                self.name,  # chip name
                self.model_name,  # model name
                self.num_chips,
                self.data_parallelism_degree,
                self.tensor_parallelism_degree,
                self.pipeline_parallelism_degree,
                self.num_data_parallel_axes,
                self.num_tensor_parallel_axes,
                self.num_pipeline_parallel_axes,
                self.data_parallel_degree_dcn,
                self.tensor_parallel_degree_dcn,
                self.pipeline_parallel_degree_dcn,
                self.microbatch_size_dcn,
                self.microbatch_size_ici,
                self.global_batch_size,
            )
        )
