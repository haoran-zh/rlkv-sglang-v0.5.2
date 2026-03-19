from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Parameter

from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.utils import set_weight_attrs


class HeadAdapterLayer(nn.Module):
    """Adapter layer with tensor parallelism support.

    The adapter is applied per attention head: adapter * o + (1 - adapter) * o_streaming
    With TP, the adapter is sharded along the head dimension.

    Args:
        num_heads: Total number of query heads (before TP split)
        num_kv_heads: Total number of key-value heads (before TP split)
        params_dtype: Data type for the parameters
        tp_rank: Tensor parallel rank
        tp_size: Tensor parallel world size
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        params_dtype: Optional[torch.dtype] = None,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ):
        super().__init__()

        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()

        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        # Calculate KV groups
        assert num_heads % num_kv_heads == 0
        self.num_kv_groups = num_heads // num_kv_heads

        # Divide heads along TP dimension
        self.tp_q_head_num = divide(num_heads, tp_size)
        self.tp_kv_head_num = divide(num_kv_heads, tp_size)

        # Create adapter parameter (one value per KV head in this partition)
        self.weight = Parameter(torch.empty(self.tp_kv_head_num, dtype=params_dtype))

        # Set weight attributes for loading
        set_weight_attrs(
            self.weight,
            {
                "output_dim": 0,  # Shard along dimension 0 (head dimension)
                "weight_loader": self.weight_loader,
            },
        )

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        """Load adapter weights with proper sharding."""
        output_dim = getattr(param, "output_dim", None)
        param_data = param.data

        # Shard the adapter weights along the head dimension
        if output_dim is not None:
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size

            # Narrow the loaded weight to this partition's slice
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        assert (
            param_data.shape == loaded_weight.shape
        ), f"Shape mismatch: {param_data.shape} vs {loaded_weight.shape}"
        param_data.copy_(loaded_weight)

    def forward(
        self,
        o: torch.Tensor,  # [batch_size, tp_q_head_num, head_dim]
        o_streaming: torch.Tensor,  # [batch_size, tp_q_head_num, head_dim]
    ) -> torch.Tensor:
        """
        Apply adapter mixing between two outputs.

        Args:
            o: Primary output tensor
            o_streaming: Streaming output tensor

        Returns:
            Mixed output: adapter * o + (1 - adapter) * o_streaming
        """

        # Expand adapter from [tp_kv_head_num] to [1, tp_q_head_num, 1]
        # Each KV head's adapter is repeated for its corresponding Q heads
        adapter = self.weight.repeat_interleave(self.num_kv_groups).view(1, -1, 1)

        # Apply adapter mixing
        o = adapter * o + (1.0 - adapter) * o_streaming

        return o

    def extra_repr(self) -> str:
        s = f"num_heads={self.num_heads}"
        s += f", num_kv_heads={self.num_kv_heads}"
        s += f", tp_q_head_num={self.tp_q_head_num}"
        s += f", tp_kv_head_num={self.tp_kv_head_num}"
        s += f", tp_size={self.tp_size}"
        s += f", num_kv_groups={self.num_kv_groups}"
        return s


class SemanticKVProjectionLayer(nn.Module):
    """Replicated low-rank projector used for semantic-KV experiments.

    The weight is intentionally replicated across tensor-parallel ranks because it is
    lightweight and the rollout-side compressed attention path is still TODO.
    """

    def __init__(
        self,
        low_rank_dim: int,
        head_dim: int,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.low_rank_dim = low_rank_dim
        self.head_dim = head_dim
        self.weight = Parameter(torch.empty(low_rank_dim, head_dim, dtype=params_dtype))
        nn.init.orthogonal_(self.weight)

    def extra_repr(self) -> str:
        return f"low_rank_dim={self.low_rank_dim}, head_dim={self.head_dim}"


def monkey_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(
        q, k, v, forward_batch, save_kv_cache=True, adapter=self.adapter
    )
    output, _ = self.o_proj(attn_output)
    return output


def monkey_qwen3_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self._apply_qk_norm(q, k)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(
        q, k, v, forward_batch, save_kv_cache=True, adapter=self.adapter
    )
    output, _ = self.o_proj(attn_output)
    return output
