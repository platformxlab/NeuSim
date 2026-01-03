import csv
from math import floor, ceil
import os
from typing import Any, Callable, Sequence

from absl import flags, logging
import numpy as np

import neusim.npusim.frontend.memory_footprint_analysis_lib as mem_footprint_lib
import neusim.npusim.frontend.Operator as Operator
import neusim.npusim.frontend.llm_ops_lib as ops_lib
import neusim.npusim.frontend.op_analysis_lib as analysis_lib
from neusim.npusim.backend import util as util
from neusim.configs.models.GLIGENConfig import GLIGENConfig


class GLIGENOpsGenerator:
    '''
    Original paper: https://arxiv.org/pdf/2301.07093.

    For UNet implementation, see https://huggingface.co/blog/annotated-diffusion.

    SinusoidalPositionEmbeddings are ignored since they account for negligible time.
    '''

    def __init__(self, config: dict[str, Any] | GLIGENConfig):
        if isinstance(config, dict):
            self.config: GLIGENConfig = GLIGENConfig.model_validate(config)
        else:
            self.config = config

        assert self.config.model_type == "gligen", f"Invalid config: {self.config}"

        self.batch_size = self.config.global_batch_size
        '''Batch size.'''
        self.noisy_latent_resolution: list[int] = self.config.unet_config.noisy_latent_resolution
        '''Input noisy latent resolution [H, W].'''
        self.image_resolution: list[int] = self.config.image_resolution
        '''Output image resolution [H, W].'''
        assert len(self.image_resolution) == 2, \
            f"Invalid image resolution: {self.image_resolution}"
        self.image_num_channels: int = self.config.image_num_channels
        '''Number of image channels.'''
        self.model_channels: int = self.config.unet_config.model_channels
        '''Model channels.'''
        self.time_embed_dim: int = self.model_channels * 4
        '''Time embedding dimension.'''
        self.context_dim: int = self.config.unet_config.context_dim
        '''Context (text grounding) model dimension.'''
        self.context_seqlen: int = self.config.grounding_input_config.text.input_seqlen
        '''Context (text grounding) sequence length.'''
        self.num_bbox = self.config.grounding_input_config.bbox.input_seqlen
        '''Number of bounding boxes.'''
        self.bbox_grounding_token_feature_dim = self.config.grounding_input_config.bbox.grounding_token_feature_dim
        '''Bounding box grounding token feature dimension.'''
        self.obj_seqlen = self.num_bbox  # TODO: need to compute this from all types of inputs
        '''Grounding object input sequence length.'''
        self.attention_resolutions: list[int] = self.config.unet_config.attention_resolutions
        '''Attention resolutions.'''
        self.num_res_blocks: int = self.config.unet_config.num_res_blocks
        '''Flattened patch size before being projected into embedding.'''
        self.channel_mult: list[int] = self.config.unet_config.channel_mult
        '''Channel multipliers.'''
        self.num_heads: int = self.config.unet_config.num_heads
        '''Number of attention heads.'''
        self.num_diffusion_steps: int = self.config.num_diffusion_steps
        '''Number of diffusion steps (each step runs an entire UNet model).'''
        self.use_flash_attention = self.config.use_flash_attention
        '''Whether to use the Flash Attention optimization.'''
        self.output_file_path: str = self.config.output_file_path
        '''Output file path.'''
        self.output_dir: str = self.config.output_dir
        '''Output directory.'''
        # os.makedirs(self.output_dir, exist_ok=True)



        # NPU configs
        self.num_sa = self.config.num_sa
        self.num_vu = self.config.num_vu
        self.hbm_bw_GBps = self.config.hbm_bw_GBps
        self.vmem_size_MB = self.config.vmem_size_MB
        self.freq_GHz = self.config.freq_GHz
        self.ici_bw_GBps = self.config.ici_bw_GBps
        '''
            Multi-chip parallelism configuration.
            TODO: Add warnings if division cannot be done cleanly.
        '''

        self.num_chips: int = self.config.num_chips
        '''Number of TPU/NPU chips'''
        self.data_parallelism: int = self.config.data_parallelism_degree
        '''Data parallelism degree.'''
        self.tensor_parallelism: int = self.config.tensor_parallelism_degree
        '''Tensor parallelism degree.'''
        self.pipeline_parallelism: int = self.config.pipeline_parallelism_degree
        '''Pipeline parallelism degree.'''
        assert(self.num_chips >= self.data_parallelism * \
            self.tensor_parallelism * self.pipeline_parallelism),\
            "Parallelism configuration is incompatible with the number of NPU chips!"

        if self.pipeline_parallelism != 1:
            raise NotImplementedError("Only pipeline parallelism of 1 is supported!")

        # if(self.batch_size % self.data_parallelism):
        #     logging.warning('''Data parallelism factor does not evenly divide batch size. Will
        #                     use ceiling division.''')
        self.batch_size = ceil(self.config.global_batch_size / self.data_parallelism)
    def generate_text_grounding_ops(self, fusion_id_start: int = 0) -> list[Operator.Operator]:
        ## TODO
        raise NotImplementedError("Not implemented.")


    def generate_bbox_grounding_ops(self, fusion_id_start: int = 0) -> list[Operator.Operator]:
        ## TODO
        raise NotImplementedError("Not implemented.")


    def generate_image_grounding_ops(self, fusion_id_start: int = 0) -> list[Operator.Operator]:
        ## TODO
        raise NotImplementedError("Not implemented.")


    def generate_keypoint_grounding_ops(self, fusion_id_start: int = 0) -> list[Operator.Operator]:
        ## TODO
        raise NotImplementedError("Not implemented.")


    def generate_spatial_condition_grounding_ops(self, fusion_id_start: int = 0) -> list[Operator.Operator]:
        ## TODO
        raise NotImplementedError("Not implemented.")

    def set_new_parallelism_cfg(self, config, t_parallel, d_parallel, p_parallel):
        self.tensor_parallelism = t_parallel
        self.data_parallelism = d_parallel
        self.pipeline_parallelism = p_parallel
        if(self.pipeline_parallelism != 1):
            raise NotImplementedError("Only pipeline parallelism of 1 is supported!")
        self.batch_size = ceil(config["batch_size"] / d_parallel)

    def Downsample(
        self,
        in_channel: int,
        scale_factor: int,
        use_conv: bool = False,
        out_channel: int | None = None,
    ) -> Callable[[Sequence[int], int], tuple[list[Operator.Operator], Sequence[int]]]:
        out_channel = out_channel or in_channel
        def block_fn(input_shape: Sequence[int], fusion_id_start: int = 0) -> tuple[list[Operator.Operator], Sequence[int]]:
            ops: list[Operator.Operator] = []
            fusion_id = fusion_id_start

            if use_conv:
                kernel_size = scale_factor + 1 if scale_factor % 2 == 0 else scale_factor
                ops += ops_lib.create_conv2d(
                    batch_size=self.batch_size,
                    input_channel=in_channel,
                    input_spatial_shape=input_shape[-2:],
                    output_channel=out_channel,
                    kernel_spatial_shape=[kernel_size, kernel_size],
                    stride=scale_factor,
                    padding=1,
                    fusion_id_start=fusion_id,
                    description_prefix=f"Downsample-Conv2d{fusion_id_start}",
                )
            else:
                ops.append(
                    ops_lib.create_downsample_op(
                        batch_size=self.batch_size,
                        input_channels=in_channel,
                        input_spatial_shape=input_shape[-2:],
                        scale_factor=scale_factor,
                        num_layers=1,
                        fusion_id_start=fusion_id,
                        description=f"Downsample{fusion_id_start}",
                    )
                )
            output_shape = [self.batch_size, out_channel, max(1, input_shape[-2] // scale_factor), max(1, input_shape[-1] // scale_factor)]
            return ops, output_shape
        return block_fn


    def Upsample(
        self,
        in_channel: int,
        scale_factor: int,
        use_conv: bool = False,
        out_channel: int | None = None,
    ) -> Callable[[Sequence[int], int], tuple[list[Operator.Operator], Sequence[int]]]:
        out_channel = out_channel or in_channel
        def block_fn(input_shape: Sequence[int], fusion_id_start: int = 0) -> tuple[list[Operator.Operator], Sequence[int]]:
            ops: list[Operator.Operator] = []
            fusion_id = fusion_id_start

            ops.append(
                ops_lib.create_upsample_op(
                    batch_size=self.batch_size,
                    input_channels=in_channel,
                    input_spatial_shape=input_shape[-2:],
                    scale_factor=scale_factor,
                    num_layers=1,
                    fusion_id_start=fusion_id,
                    description=f"Upsample{fusion_id_start}",
                )
            )
            if use_conv:
                kernel_size = scale_factor + 1 if scale_factor % 2 == 0 else scale_factor
                ops += ops_lib.create_conv2d(
                    batch_size=self.batch_size,
                    input_channel=in_channel,
                    input_spatial_shape=input_shape[-2:],
                    output_channel=out_channel,
                    kernel_spatial_shape=[kernel_size, kernel_size],
                    padding=1,
                    fusion_id_start=fusion_id,
                    description_prefix=f"Upsample-Conv2d{fusion_id_start}",
                )
            output_shape = [self.batch_size, out_channel, input_shape[-2] * scale_factor, input_shape[-1] * scale_factor]
            return ops, output_shape
        return block_fn


    def Convolution(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> Callable[[Sequence[int], int], tuple[list[Operator.Operator], Sequence[int]]]:
        def compute_output_spatial_dim_size(in_dim, padding, dilation, kernel, stride) -> int:
            ## see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            return floor((in_dim + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)

        def block_fn(input_shape: Sequence[int], fusion_id_start: int = 0) -> tuple[list[Operator.Operator], Sequence[int]]:
            input_spatial_shape = input_shape[-2:]
            kernel_spatial_shape = [kernel_size, kernel_size]
            output_spatial_shape = [
                compute_output_spatial_dim_size(input_spatial_shape[0], padding, dilation, kernel_spatial_shape[0], stride),
                compute_output_spatial_dim_size(input_spatial_shape[1], padding, dilation, kernel_spatial_shape[1], stride),
            ]
            output_shape = [self.batch_size, out_channel, *output_spatial_shape]

            ops: list[Operator.Operator] = []
            fusion_id = fusion_id_start
            ops += ops_lib.create_conv2d(
                batch_size=self.batch_size,
                input_channel=in_channel,
                input_spatial_shape=input_spatial_shape,
                output_channel=out_channel,
                kernel_spatial_shape=kernel_spatial_shape,
                stride=stride,
                padding=padding,
                dilation=dilation,
                fusion_id_start=fusion_id,
                description_prefix=f"Conv2d{fusion_id_start}",
            )

            return ops, output_shape
        return block_fn


    def Block(
        self,
        dim: int,
        dim_out: int,
        groups: int = 8,
        up: bool = False,
        down: bool = False,
    ) -> Callable[[Sequence[int], int], tuple[list[Operator.Operator], Sequence[int]]]:
        assert not (up and down), "Invalid config: up and down cannot be True simultaneously."
        def block_fn(input_shape: Sequence[int], fusion_id_start: int = 0) -> tuple[list[Operator.Operator], Sequence[int]]:
            ops: list[Operator.Operator] = []
            fusion_id = fusion_id_start

            if up:
                ops.append(
                    ops_lib.create_upsample_op(
                        batch_size=self.batch_size,
                        input_channels=dim,
                        input_spatial_shape=input_shape[-2:],
                        scale_factor=2,
                        num_layers=1,
                        fusion_id_start=fusion_id,
                        description=f"x-Upsample{fusion_id_start}",
                    )
                )
                fusion_id += 1
            elif down:
                ops.append(
                    ops_lib.create_downsample_op(
                        batch_size=self.batch_size,
                        input_channels=dim,
                        input_spatial_shape=input_shape[-2:],
                        scale_factor=2,
                        num_layers=1,
                        fusion_id_start=fusion_id,
                        description=f"x-Downsample{fusion_id_start}",
                    )
                )
                fusion_id += 1
            ops.append(
                ops_lib.create_unary_op(
                    input_shape=input_shape,
                    op_name="GroupNorm",
                    name="X_norm = GroupNorm(X)",
                    description=f"Conv2d-GroupNorm{fusion_id_start}",
                    count=1,
                    fusion_id=fusion_id,
                )
            )
            fusion_id += 1
            if up:
                ops.append(
                    ops_lib.create_upsample_op(
                        batch_size=self.batch_size,
                        input_channels=dim,
                        input_spatial_shape=input_shape[-2:],
                        scale_factor=2,
                        num_layers=1,
                        fusion_id_start=fusion_id,
                        description=f"h-Upsample{fusion_id_start}",
                    )
                )
                fusion_id += 1
            elif down:
                ops.append(
                    ops_lib.create_downsample_op(
                        batch_size=self.batch_size,
                        input_channels=dim,
                        input_spatial_shape=input_shape[-2:],
                        scale_factor=2,
                        num_layers=1,
                        fusion_id_start=fusion_id,
                        description=f"h-Downsample{fusion_id_start}",
                    )
                )
                fusion_id += 1
            ops += ops_lib.create_conv2d(
                batch_size=self.batch_size,
                input_channel=dim,
                input_spatial_shape=input_shape[-2:],
                output_channel=dim_out,
                kernel_spatial_shape=[3, 3],
                padding=1,
                fusion_id_start=fusion_id,
                description_prefix=f"Conv2d{fusion_id_start}",
            )
            ### Ignored: scale-shift and SiLU activation (assume fused with previous/next op)

            if up:
                output_spatial_shape = [input_shape[-2] * 2, input_shape[-1] * 2]
            elif down:
                output_spatial_shape = [max(1, input_shape[-2] // 2), max(1, input_shape[-1] // 2)]
            else:
                output_spatial_shape = input_shape[-2:]
            output_shape = [self.batch_size, dim_out, *output_spatial_shape]
            return ops, output_shape
        return block_fn


    def ResnetBlock(
        self,
        dim: int,
        time_emb_dim: int,
        dim_out: int | None = None,
        groups: int = 8,
        up: bool = False,
        down: bool = False,
    ) -> Callable[[Sequence[int], int], tuple[list[Operator.Operator], Sequence[int]]]:
        dim_out = dim_out or dim
        block1 = self.Block(dim, dim_out, groups, up, down)
        block2 = self.Block(dim_out, dim_out, groups, False, False)

        def block_fn(input_shape: Sequence[int], fusion_id_start: int = 0) -> tuple[list[Operator.Operator], Sequence[int]]:
            ops: list[Operator.Operator] = []
            fusion_id = fusion_id_start

            # Time Embedding MLP block
            ops.append(
                # Ignored: SiLU activation (assume fused with MatMul op)
                ops_lib.create_einsum_op(
                    input_a_shape=[self.batch_size, time_emb_dim],
                    input_b_shape=[time_emb_dim, dim_out],
                    einsum_expr="BT;TD->BD",
                    dtype="DT_BFLOAT16",
                    memory_placement=[0, 0, 0],
                    fusion_id=fusion_id,
                    description=f"Time-Embed-MLP-Einsum{fusion_id_start}",
                    name="einsum",
                    count=1,
                )
            )
            ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype("DT_BFLOAT16") \
                                    * int(np.prod([time_emb_dim, dim_out]))
            fusion_id += 1
            new_ops, output_shape = block1(input_shape, fusion_id)
            ops += new_ops
            fusion_id = ops[-1].fusion_id + 1
            new_ops, output_shape = block2(output_shape, fusion_id)
            ops += new_ops
            fusion_id = ops[-1].fusion_id + 1
            # skip connection
            if dim != dim_out:
                ops.append(
                    ops_lib.create_einsum_op(
                    input_a_shape=[self.batch_size, *input_shape[-2:], dim],
                    input_b_shape=[dim, dim_out],
                    einsum_expr="BHWC;CO->BHWO",
                    dtype="DT_BFLOAT16",
                    memory_placement=[0, 0, 0],
                    fusion_id=fusion_id,
                    description=f"SkipConnection-Einsum{fusion_id_start}",
                    name="einsum",
                    count=1,
                    )
                )
                ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype("DT_BFLOAT16") \
                        * int(np.prod([dim, dim_out]))
                output_shape = [self.batch_size, dim_out, *output_shape[-2:]]

            return ops, output_shape
        return block_fn


    def SelfAttentionBlock(
        self,
        d_query: int,
        num_heads: int,
        d_head: int,
    ) -> Callable[[int, int], tuple[list[Operator.Operator], Sequence[int]]]:
        '''
        class SelfAttention(nn.Module):
            def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
                super().__init__()
                inner_dim = dim_head * heads
                self.scale = dim_head ** -0.5
                self.heads = heads

                self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
                self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
                self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

                self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

            def forward(self, x):
                q = self.to_q(x) # B*N*(H*C)
                k = self.to_k(x) # B*N*(H*C)
                v = self.to_v(x) # B*N*(H*C)

                B, N, HC = q.shape
                H = self.heads
                C = HC // H

                q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
                k = k.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
                v = v.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C

                sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
                attn = sim.softmax(dim=-1) # (B*H)*N*N

                out = torch.einsum('b i j, b j c -> b i c', attn, v) # (B*H)*N*C
                out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

                return self.to_out(out)
        '''
        def block_fn(input_seqlen: int, fusion_id_start: int = 0) -> tuple[list[Operator.Operator], Sequence[int]]:
            ops: list[Operator.Operator] = []
            fusion_id = fusion_id_start

            ops += ops_lib.create_multi_head_attention(
                batch_size=self.batch_size,
                input_seqlen=input_seqlen,
                output_seqlen=1,  # unused
                decode_width=1,  # unused
                num_heads=num_heads,
                d_model=d_query,
                d_head=d_head,
                config=self.config,
                num_layers=1,
                fusion_id_start=fusion_id,
                description_prefix=f"SelfAttention{fusion_id_start}",
                use_flash_attention=self.use_flash_attention,
                tensor_parallelism_axes=[self.tensor_parallelism],  # TODO: support multi-axis
                ici_bw_GBps=self.ici_bw_GBps
            )
            output_shape = [self.batch_size, input_seqlen, d_query]
            return ops, output_shape
        return block_fn


    def CrossAttentionBlock(
        self,
        d_query: int,
        d_key: int,
        d_value: int,
        num_heads: int,
        d_head: int,
    ) -> Callable[[int, int, int], tuple[list[Operator.Operator], Sequence[int]]]:
        '''
        class CrossAttention(nn.Module):
            def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0):
                super().__init__()
                inner_dim = dim_head * heads
                self.scale = dim_head ** -0.5
                self.heads = heads

                self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
                self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
                self.to_v = nn.Linear(value_dim, inner_dim, bias=False)

                self.to_out = nn.Sequential( nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

            def forward(self, x, key, value, mask=None):

                q = self.to_q(x)     # B*N*(H*C)
                k = self.to_k(key)   # B*M*(H*C)
                v = self.to_v(value) # B*M*(H*C)

                B, N, HC = q.shape
                _, M, _ = key.shape
                H = self.heads
                C = HC // H

                q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
                k = k.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C
                v = v.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C

                sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale # (B*H)*N*M
                attn = sim.softmax(dim=-1) # (B*H)*N*M

                out = torch.einsum('b i j, b j d -> b i d', attn, v) # (B*H)*N*C
                out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

                return self.to_out(out)
        '''
        def block_fn(query_seqlen: int, kv_seqlen: int, fusion_id_start: int = 0) -> tuple[list[Operator.Operator], Sequence[int]]:
            ops: list[Operator.Operator] = []
            fusion_id = fusion_id_start

            # ops += ops_lib.create_multi_head_cross_attention(
            #     batch_size=self.batch_size,
            #     q_seqlen=query_seqlen,
            #     kv_seqlen=kv_seqlen,
            #     num_heads=num_heads,
            #     d_query=d_query,
            #     d_key=d_key,
            #     d_value=d_value,
            #     d_head=d_head,
            #     num_layers=1,
            #     dtype="DT_BFLOAT16",
            #     fusion_id_start=fusion_id,
            #     description_prefix=f"CrossAttention{fusion_id_start}",
            #     use_flash_attention=self.use_flash_attention,
            #     tensor_parallelism_degree=self.tensor_parallelism,
            #     ici_bw_GBps=self.ici_bw_GBps
            # )

            # Cross Attention -- Args with values of -1 are unused.
            ops += ops_lib.create_multi_head_attention(
                batch_size=self.batch_size,
                input_seqlen= -1,
                output_seqlen= -1,
                decode_width = -1,
                d_model= -1,
                num_heads=num_heads,
                d_query=d_query,
                config=self.config,
                num_layers=1,
                dtype="DT_BFLOAT16",
                fusion_id_start=fusion_id,
                description_prefix=f"CrossAttention{fusion_id_start}",
                type="cross-attention",
                q_seqlen=query_seqlen,
                kv_seqlen=kv_seqlen,
                d_key=d_key,
                d_value=d_value,
                d_head=d_head,
                use_flash_attention=self.use_flash_attention,
                tensor_parallelism_axes=[self.tensor_parallelism],  # TODO: support multi-axis
                ici_bw_GBps=self.ici_bw_GBps
            )
            output_shape = [self.batch_size, query_seqlen, d_query]
            return ops, output_shape
        return block_fn


    def GatedSelfAttentionBlock(
        self,
        d_query: int,
        d_context: int,
        num_heads: int,
        d_head: int,
    ) -> Callable[[int, int, int], tuple[list[Operator.Operator], Sequence[int]]]:
        '''
        class GatedSelfAttentionDense(nn.Module):
            def __init__(self, query_dim, context_dim,  n_heads, d_head):
                super().__init__()

                # we need a linear projection since we need cat visual feature and obj feature
                self.linear = nn.Linear(context_dim, query_dim)

                self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
                self.ff = FeedForward(query_dim, glu=True)

                self.norm1 = nn.LayerNorm(query_dim)
                self.norm2 = nn.LayerNorm(query_dim)

                self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
                self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

                # this can be useful: we can externally change magnitude of tanh(alpha)
                # for example, when it is set to 0, then the entire model is same as original one
                self.scale = 1


            def forward(self, x, objs):

                N_visual = x.shape[1]
                objs = self.linear(objs)

                x = x + self.scale*torch.tanh(self.alpha_attn) * self.attn(  self.norm1(torch.cat([x,objs],dim=1))  )[:,0:N_visual,:]
                x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) )

                return x
        '''
        def block_fn(query_seqlen: int, kv_seqlen: int, fusion_id_start: int = 0) -> tuple[list[Operator.Operator], Sequence[int]]:
            ops: list[Operator.Operator] = []
            fusion_id = fusion_id_start

            # Linear projection to align context and query dim
            ops.append(
                ops_lib.create_einsum_op(
                    input_a_shape=[self.batch_size, kv_seqlen, d_context],
                    input_b_shape=[d_context, d_query],
                    einsum_expr="BLM;MD->BLD",
                    name="X = Linear(context)",
                    description=f"GatedSelfAttention-Linear{fusion_id}",
                    count=1,
                    fusion_id=fusion_id,
                )
            )
            ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype("DT_BFLOAT16") \
                                    * int(np.prod([d_context, d_query]))
            fusion_id += 1
            ops += ops_lib.create_multi_head_attention(
                batch_size=self.batch_size,
                input_seqlen=(query_seqlen + kv_seqlen),
                output_seqlen=1,  # unused
                decode_width=1,  # unused
                num_heads=num_heads,
                d_model=d_query,
                d_head=d_head,
                config=self.config,
                num_layers=1,
                dtype="DT_BFLOAT16",
                fusion_id_start=fusion_id,
                description_prefix=f"GatedSelfAttention-Attn{fusion_id_start}",
                use_flash_attention=self.use_flash_attention,
                tensor_parallelism_axes=[self.tensor_parallelism],  # TODO: support multi-axis
                ici_bw_GBps=self.ici_bw_GBps
            )
            fusion_id = ops[-1].fusion_id + 1
            # FFN
            ops += ops_lib.create_ffn(
                batch_size=self.batch_size,
                input_seqlen=query_seqlen,
                output_seqlen=1,  # unused
                decode_width=1,  # unused
                d_model=d_query,
                d_ff=d_query * 4,
                config=self.config,
                num_layers=1,
                ffn_type="default",
                fusion_id_start=fusion_id,
                description_prefix=f"GatedSelfAttention-FFN{fusion_id_start}",
                tensor_parallelism_axes=[self.tensor_parallelism],  # TODO: support multi-axis
                ici_bw_GBps=self.ici_bw_GBps
            )
            output_shape = [self.batch_size, query_seqlen, d_query]
            return ops, output_shape
        return block_fn


    def BasicTransformerBlock(
        self,
        d_query: int,
        d_key: int,
        d_value: int,
        num_heads: int,
        d_head: int,
    ) -> Callable[[int, int], tuple[list[Operator.Operator], Sequence[int]]]:
        '''
        class BasicTransformerBlock(nn.Module):
            def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head):
                super().__init__()
                self.attn1 = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
                self.ff = FeedForward(query_dim, glu=True)
                self.attn2 = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head)
                self.norm1 = nn.LayerNorm(query_dim)
                self.norm2 = nn.LayerNorm(query_dim)
                self.norm3 = nn.LayerNorm(query_dim)

                # note key_dim here actually is context_dim
                self.fuser = GatedSelfAttentionDense(query_dim, key_dim, n_heads, d_head)

            def _forward(self, x, context, objs):
                x = self.attn1( self.norm1(x) ) + x
                x = self.fuser(x, objs) # identity mapping in the beginning
                x = self.attn2(self.norm2(x), context, context) + x
                x = self.ff(self.norm3(x)) + x
                return x
        '''
        attn1 = self.SelfAttentionBlock(d_query, num_heads, d_head)
        attn2 = self.CrossAttentionBlock(d_query, d_key, d_value, num_heads, d_head)
        fuser = self.GatedSelfAttentionBlock(d_query, d_key, num_heads, d_head)

        def block_fn(query_seqlen: int, fusion_id_start: int = 0) -> tuple[list[Operator.Operator], Sequence[int]]:
            ops: list[Operator.Operator] = []
            fusion_id = fusion_id_start

            ops.append(
                ops_lib.create_unary_op(
                    input_shape=[self.batch_size, query_seqlen, d_query],
                    op_name="LayerNorm",
                    name="X_norm = LayerNorm(X)",
                    description=f"BasicTransformerBlock-Input_layernorm{fusion_id}",
                    count=1,
                    fusion_id=fusion_id,
                )
            )
            fusion_id += 1
            # Ignored: all residual connections
            new_ops, output_shape = attn1(query_seqlen, fusion_id)
            ops += new_ops
            fusion_id = ops[-1].fusion_id + 1
            new_ops, output_shape = fuser(query_seqlen, self.obj_seqlen, fusion_id)
            ops += new_ops
            fusion_id = ops[-1].fusion_id + 1
            ops.append(
                ops_lib.create_unary_op(
                    input_shape=[self.batch_size, query_seqlen, d_query],
                    op_name="LayerNorm",
                    name="X_norm = LayerNorm(X)",
                    description=f"BasicTransformerBlock-Fuser_output_layernorm{fusion_id}",
                    count=1,
                    fusion_id=fusion_id,
                )
            )
            fusion_id += 1
            new_ops, output_shape = attn2(query_seqlen, self.context_seqlen, fusion_id)
            ops += new_ops
            fusion_id = ops[-1].fusion_id + 1
            ops.append(
                ops_lib.create_unary_op(
                    input_shape=[self.batch_size, query_seqlen, d_query],
                    op_name="LayerNorm",
                    name="X_norm = LayerNorm(X)",
                    description=f"BasicTransformerBlock-Attn_output_layernorm{fusion_id}",
                    count=1,
                    fusion_id=fusion_id,
                )
            )
            fusion_id += 1
            ops += ops_lib.create_ffn(
                batch_size=self.batch_size,
                input_seqlen=query_seqlen,
                output_seqlen=1,  # unused
                decode_width=1,  # unused
                d_model=d_query,
                d_ff=d_query * 4,
                config=self.config,
                num_layers=1,
                ffn_type="default",
                fusion_id_start=fusion_id,
                description_prefix=f"BasicTransformerBlock-FFN{fusion_id}",
                tensor_parallelism_axes=[self.tensor_parallelism],  # TODO: support multi-axis
                ici_bw_GBps=self.ici_bw_GBps
            )

            output_shape = [self.batch_size, query_seqlen, d_query]
            return ops, output_shape
        return block_fn


    def SpatialTransformerBlock(
        self,
        num_channels: int,
        d_key: int,
        d_value: int,
        num_heads: int,
        d_head: int,
    ) -> Callable[[Sequence[int], int], tuple[list[Operator.Operator], Sequence[int]]]:
        '''
        class SpatialTransformer(nn.Module):
            def __init__(self, in_channels, key_dim, value_dim, n_heads, d_head, depth=1):
                super().__init__()
                self.in_channels = in_channels
                query_dim = n_heads * d_head
                self.norm = Normalize(in_channels)

                self.proj_in = nn.Conv2d(in_channels, query_dim, kernel_size=1, stride=1, padding=0)

                self.transformer_blocks = nn.ModuleList(
                    [BasicTransformerBlock(query_dim, key_dim, value_dim, n_heads, d_head)
                        for d in range(depth)]
                )

                self.proj_out = nn.Conv2d(query_dim, in_channels, kernel_size=1, stride=1, padding=0)

            def forward(self, x, context, objs):
                b, c, h, w = x.shape
                x_in = x
                x = self.norm(x)
                x = self.proj_in(x)
                x = rearrange(x, 'b c h w -> b (h w) c')
                for block in self.transformer_blocks:
                    x = block(x, context, objs)
                x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
                x = self.proj_out(x)
                return x + x_in
        '''
        d_query = num_heads * d_head
        transformer_block = self.BasicTransformerBlock(d_query, d_key, d_value, num_heads, d_head)

        def block_fn(input_shape: Sequence[int], fusion_id_start: int = 0) -> tuple[list[Operator.Operator], Sequence[int]]:
            ops: list[Operator.Operator] = []
            fusion_id = fusion_id_start
            input_seqlen = input_shape[-2] * input_shape[-1]

            ops.append(
                ops_lib.create_unary_op(
                    input_shape=input_shape,
                    op_name="GroupNorm",
                    name="X_norm = GroupNorm(X)",
                    description=f"SpatialTransformer-Input_GroupNorm{fusion_id}",
                    count=1,
                    fusion_id=fusion_id,
                )
            )
            fusion_id += 1
            ops.append(
                ops_lib.create_einsum_op(
                    input_a_shape=[self.batch_size, input_seqlen, num_channels],
                    input_b_shape=[num_channels, d_query],
                    einsum_expr="BSN;NC->BSC",
                    dtype="DT_BFLOAT16",
                    memory_placement=[0, 0, 0],
                    fusion_id=fusion_id,
                    description=f"SpatialTransformer-Proj_in{fusion_id}",
                )
            )
            ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype("DT_BFLOAT16")\
                                    * int(np.prod([num_channels, d_query]))
            fusion_id += 1
            new_ops, output_shape = transformer_block(input_seqlen, fusion_id)
            ops += new_ops
            fusion_id = ops[-1].fusion_id + 1
            ops.append(
                ops_lib.create_einsum_op(
                    input_a_shape=[self.batch_size, input_seqlen, d_query],
                    input_b_shape=[d_query, num_channels],
                    einsum_expr="BSN;NC->BSC",
                    dtype="DT_BFLOAT16",
                    memory_placement=[0, 0, 0],
                    fusion_id=fusion_id,
                    description=f"SpatialTransformer-Proj_out{fusion_id}",
                )
            )
            ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype("DT_BFLOAT16")\
                        * int(np.prod([num_channels, d_query]))
            output_shape = input_shape
            return ops, output_shape
        return block_fn


    def generate_time_embedding_ops(self, fusion_id_start: int = 0) -> list[Operator.Operator]:
        ops: list[Operator.Operator] = []
        fusion_id = fusion_id_start
        model_ch_parallel = ceil(self.model_channels / self.tensor_parallelism)
        time_embed_dim_parallel = ceil(self.time_embed_dim / self.tensor_parallelism)
        ops.append(
            ops_lib.create_einsum_op(
                input_a_shape=[self.batch_size, model_ch_parallel],
                input_b_shape=[model_ch_parallel, self.time_embed_dim],
                einsum_expr="BT;TD->BD",
                dtype="DT_BFLOAT16",
                memory_placement=[0, 0, 0],
                fusion_id=fusion_id,
                description=f"Time-Embed-MLP-FFi{fusion_id_start}",
            )
        )
        ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype("DT_BFLOAT16") \
                                * int(np.prod([self.model_channels, self.time_embed_dim]))
        fusion_id += 1
        ops.append(
            ops_lib.create_einsum_op(
                input_a_shape=[self.batch_size, time_embed_dim_parallel],
                input_b_shape=[time_embed_dim_parallel, self.time_embed_dim],
                einsum_expr="BT;TD->BD",
                dtype="DT_BFLOAT16",
                memory_placement=[0, 0, 0],
                fusion_id=fusion_id,
                description=f"Time-Embed-MLP-FFo{fusion_id_start}",
            )
        )
        ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype("DT_BFLOAT16") \
                                * int(np.prod([time_embed_dim_parallel, self.time_embed_dim]))

        return ops


    def generate_UNet_ops(self, fusion_id_start: int = 2) -> list[Operator.Operator]:
        ############################# Down Branch #############################
        # [(has context input, block_fn), ...]
        down_branch_blocks: list[list[Callable[[Sequence[int], int], tuple[list[Operator.Operator], Sequence[int]]]]] = [
            [self.Convolution(self.image_num_channels, self.model_channels, 3, padding=1)]
        ]
        down_channels: list[int] = [self.model_channels]
        ch = self.model_channels
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = []
                layers.append(
                    self.ResnetBlock(
                        dim=ch,
                        time_emb_dim=self.time_embed_dim,
                        dim_out=(mult * self.model_channels),
                    )
                )
                ch = mult * self.model_channels
                if ds in self.attention_resolutions:
                    dim_head = ch // self.num_heads
                    layers.append(
                        self.SpatialTransformerBlock(
                            num_channels=ch,
                            d_key=self.context_dim,
                            d_value=self.context_dim,
                            num_heads=self.num_heads,
                            d_head=dim_head,
                        )
                    )
                down_channels.append(ch)
                down_branch_blocks.append(layers)
            if level != len(self.channel_mult) - 1: # will not go to this downsample branch in the last feature
                out_ch = ch
                down_branch_blocks.append([
                    self.Downsample(
                        in_channel=ch,
                        scale_factor=2,
                        use_conv=True,
                        out_channel=out_ch,
                    )
                ])
                ch = out_ch
                down_channels.append(ch)
                ds *= 2

        dim_head = ch // self.num_heads
        ###### down_branch_blocks = [ C |  RT  RT  D  |  RT  RT  D  |  RT  RT  D  |   R  R   ] #########

        ############################# Middle Bottleneck #############################
        middle_blocks: list[Callable[[Sequence[int], int], tuple[list[Operator.Operator], Sequence[int]]]] = [
            self.ResnetBlock(
                dim=ch,
                time_emb_dim=self.time_embed_dim,
            ),
            self.SpatialTransformerBlock(
                num_channels=ch,
                d_key=self.context_dim,
                d_value=self.context_dim,
                num_heads=self.num_heads,
                d_head=dim_head,
            ),
            self.ResnetBlock(
                dim=ch,
                time_emb_dim=self.time_embed_dim,
            ),
        ]
        ###### middle_blocks = [ RTR ] #########

        ############################## Up Branch ##############################
        up_branch_blocks: list[list[Callable[[Sequence[int], int], tuple[list[Operator.Operator], Sequence[int]]]]] = []
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            for i in range(self.num_res_blocks + 1):
                ich = down_channels.pop()
                layers = []
                layers.append(
                    self.ResnetBlock(
                        dim=(ch + ich),
                        time_emb_dim=self.time_embed_dim,
                        dim_out=(self.model_channels * mult),
                    )
                )
                ch = self.model_channels * mult

                if ds in self.attention_resolutions:
                    dim_head = ch // self.num_heads
                    layers.append(
                        self.SpatialTransformerBlock(
                            num_channels=ch,
                            d_key=self.context_dim,
                            d_value=self.context_dim,
                            num_heads=self.num_heads,
                            d_head=dim_head,
                        )
                    )
                if level and i == self.num_res_blocks:
                    out_ch = ch
                    layers.append(
                        self.Upsample(
                            in_channel=ch,
                            scale_factor=2,
                            use_conv=True,
                            out_channel=out_ch,
                        )
                    )
                    ds //= 2
                up_branch_blocks.append(layers)
        ###### up_branch_blocks = [ R  R  RU | RT  RT  RTU |  RT  RT  RTU  |  RT  RT  RT   ] #########

        ############################# Generate Ops ############################
        ops: list[Operator.Operator] = []
        fusion_id = fusion_id_start

        input_shape = [self.batch_size, self.image_num_channels, *self.noisy_latent_resolution]

        # Time embedding ops
        ops += self.generate_time_embedding_ops(fusion_id)
        fusion_id = ops[-1].fusion_id + 1

        # TODO: Make sure first chip does time embed instead of input xfer.
        if(self.pipeline_parallelism > 1):
            ops.append(ops_lib.create_input_transfer_op(
                input_shape= [self.batch_size, self.time_embed_dim],
                config=self.config,
                fusion_id_start= fusion_id,
                description= "Receive results from previous pipeline stage",
                name= "PipelineInputOp",
                count = 1
            ))
        fusion_id += 1

        for _ in range(self.num_diffusion_steps):
            down_residual_shapes = []
            for modules in down_branch_blocks:
                for module in modules:
                    new_ops, input_shape = module(input_shape, fusion_id)
                    ops += new_ops
                    fusion_id = ops[-1].fusion_id + 1
                down_residual_shapes.append(input_shape)

            for module in middle_blocks:
                new_ops, input_shape = module(input_shape, fusion_id)
                ops += new_ops
                fusion_id = ops[-1].fusion_id + 1

            for modules in up_branch_blocks:
                residual_shape = down_residual_shapes.pop()
                # Concat the image channel dimension.
                # This actually has no effect in our code right now...
                # because the channel concatenation is done during layer creation (ch + ich) in the up branch
                input_shape = [
                    self.batch_size,
                    input_shape[1] + residual_shape[1],
                    *[max(x, y) for x, y in zip(input_shape[2:], residual_shape[2:])]
                ]
                for module in modules:
                    new_ops, input_shape = module(input_shape, fusion_id)
                    ops += new_ops
                    fusion_id = ops[-1].fusion_id + 1

            # final group norm and conv2d
            ops.append(
                ops_lib.create_unary_op(
                    input_shape=input_shape,
                    op_name="GroupNorm",
                    name="X_norm = GroupNorm(X)",
                    description=f"Out{fusion_id}-GroupNorm",
                    count=1,
                    fusion_id=fusion_id,
                )
            )
            fusion_id += 1
            ops += ops_lib.create_conv2d(
                batch_size=self.batch_size,
                input_channel=self.model_channels,
                input_spatial_shape=input_shape[-2:],
                output_channel=self.image_num_channels,
                kernel_spatial_shape=[3, 3],
                padding=1,
                fusion_id_start=fusion_id,
                description_prefix=f"Out{fusion_id}-",
            )
            fusion_id = ops[-1].fusion_id + 1

        if self.pipeline_parallelism > 1:
            ops.append(ops_lib.create_output_transfer_op(
                input_shape=input_shape,
                config=self.config,
                fusion_id_start= fusion_id,
                description="Pass results to next pipeline stage",
                name="PipelineOutputOp",
                count=1,
            ))

        return ops

    '''
    TODO: Add interchip communication stats to CSV.
    '''
    def generate(self, fusion_id_start: int = 2, dump_to_file: bool = False, dump_converted_node_costs: bool = False, analyze_energy: bool = True) -> list[Operator.Operator]:
        ops: list[Operator.Operator] = []
        fusion_id = fusion_id_start

        ops += self.generate_UNet_ops(fusion_id)
        ops = analysis_lib.fill_operators_execution_info(
            ops, self.config, analyze_energy=analyze_energy
        )

        if dump_to_file:
            logging.info(
                "Generating GLIGEN ops and dumping to %s.", os.path.abspath(self.output_file_path)
            )
            ops_dict = [Operator.to_csv_dict(op) for op in ops]
            with open(self.output_file_path, "w") as f:
                writer = csv.DictWriter(f, fieldnames=ops_dict[0].keys())
                writer.writeheader()
                writer.writerows(ops_dict)
        if dump_converted_node_costs:
            ops_dict = [Operator.to_csv_dict(op) for op in ops]
            self.parse_tf_sim_analytical_traces(ops_dict, self.output_dir)

        return ops


    def parse_tf_sim_analytical_traces(self, ops: list[dict[str, Any]], output_dir: str | None = None) -> list[dict[str, Any]]:
        converted_ops = []
        for op in ops:
            converted_op= {
                "Op Name": op["Op Name"],
                "Top Level Node": True,
                "Op Code": op["Op Code"],
                "Total execution time (ns)": op["Execution time"],
                "Execution Time (ns)": op["Execution time"],
                "Compute Time (ns)": op["Compute time"],
                "Memory Time (ns)": op["Memory time"],
                "Bound By": op["Bounded-by"] if op["Bounded-by"] == "Compute" else "External Memory",
                "Total MXU time (ns)": op["MXU time"],
                "Total VPU time (ns)": op["VPU time"],
                "ICI/NVLink time": op["ICI/NVLink time"],
                "ICI/NVLink outbound traffic": op["ICI/NVLink outbound traffic"],
                "ICI/NVLink inbound traffic": op["ICI/NVLink inbound traffic"],
                "Transpose time": op["Transpose time"],
                "Permute time": op["Permute time"],
                "Bytes Accessed": op["Bytes accessed"],
                "FLOP Count": op["FLOP Count"],
                "Input Tensor Shapes": op["Input Tensor Shapes"],
                "Output Tensor Shapes": op["Output Tensor Shapes"],
                "Config": op["Config"],
            }
            converted_ops.append(converted_op)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.dump_ops_to_csv(converted_ops, os.path.join(output_dir, "node_costs.csv"))

        return converted_ops


    def dump_ops_to_csv(self, ops: list[Operator.Operator], output_path: str):
        logging.info("Dumping ops to %s.", os.path.abspath(output_path))
        ops_dict = [Operator.to_csv_dict(op) for op in ops]
        with open(output_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=ops_dict[0].keys())
            writer.writeheader()
            writer.writerows(ops_dict)


    def compute_memory_footprint_bytes(self) -> int:
        '''
        Compute the memory footprint in bytes.
        '''
        return mem_footprint_lib.get_gligen_inference_mem_requirement(self.config)
