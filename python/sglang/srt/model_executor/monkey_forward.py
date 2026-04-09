from typing import Dict, List, Optional, Set, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.utils import set_weight_attrs


def _init_orthogonal_param(param: Parameter):
    with torch.no_grad():
        init_weight = torch.empty(
            param.shape,
            device=param.device,
            dtype=torch.float32,
        )
        nn.init.orthogonal_(init_weight)
        param.copy_(init_weight.to(dtype=param.dtype))


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


def _tensor_to_slot_set(slots: Optional[torch.Tensor]) -> Set[int]:
    if slots is None or slots.numel() == 0:
        return set()
    return {int(slot) for slot in slots.detach().view(-1).cpu().tolist() if int(slot) > 0}


class SemanticKVCompactionManager:
    """Track per-request semantic-KV ownership and reclaim dead token slots.

    Token slots are shared across all layers in SGLang's allocator, while semantic-KV
    selection is layer-specific. We therefore keep per-layer retained slot sets and a
    per-request refcount over physical token slots. A slot is returned to the allocator
    only after every semantic-KV layer has dropped it.
    """

    def __init__(self, allocator, num_layers: int):
        self.allocator = allocator
        self.num_layers = num_layers
        self.layer_slots: Dict[str, Dict[int, Set[int]]] = {}
        self.slot_ref_counts: Dict[str, Dict[int, int]] = {}
        self.slot_observers: Dict[str, Dict[int, Set[int]]] = {}

    def _free_slots(self, slots: Set[int]):
        if not slots:
            return
        free_tensor = torch.tensor(
            sorted(slots),
            dtype=torch.int64,
            device=self.allocator.device,
        )
        self.allocator.free(free_tensor)

    def reset_request(self, request_key: str, free_slots: bool = True):
        ref_counts = self.slot_ref_counts.pop(request_key, {})
        observers = self.slot_observers.pop(request_key, {})
        self.layer_slots.pop(request_key, None)

        if free_slots:
            self._free_slots(set(ref_counts.keys()) | set(observers.keys()))

    def update_layer_slots(
        self,
        request_key: str,
        layer_id: int,
        new_slots: torch.Tensor,
        observed_slots: Optional[torch.Tensor] = None,
    ):
        new_slot_set = _tensor_to_slot_set(new_slots)
        request_layer_slots = self.layer_slots.setdefault(request_key, {})
        request_ref_counts = self.slot_ref_counts.setdefault(request_key, {})
        old_slot_set = request_layer_slots.get(layer_id, set())

        for slot in old_slot_set - new_slot_set:
            count = request_ref_counts.get(slot, 0) - 1
            if count <= 0:
                request_ref_counts.pop(slot, None)
                observed = self.slot_observers.get(request_key, {}).get(slot)
                if observed is None or len(observed) >= self.num_layers:
                    self._free_slots({slot})
            else:
                request_ref_counts[slot] = count

        for slot in new_slot_set - old_slot_set:
            request_ref_counts[slot] = request_ref_counts.get(slot, 0) + 1

        request_layer_slots[layer_id] = new_slot_set

        if observed_slots is None:
            return

        request_observers = self.slot_observers.setdefault(request_key, {})
        for slot in _tensor_to_slot_set(observed_slots):
            observed = request_observers.setdefault(slot, set())
            observed.add(layer_id)
            if len(observed) >= self.num_layers:
                request_observers.pop(slot, None)
                if request_ref_counts.get(slot, 0) <= 0:
                    self._free_slots({slot})

        if not request_observers:
            self.slot_observers.pop(request_key, None)

    def get_live_slots(self, request_key: str) -> List[int]:
        return sorted(self.slot_ref_counts.get(request_key, {}).keys())

    def get_live_slots_tensor(self, request_key: str, device) -> torch.Tensor:
        live_slots = self.get_live_slots(request_key)
        if not live_slots:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.tensor(live_slots, dtype=torch.long, device=device)

    def get_layer_slots_tensor(
        self,
        request_key: str,
        layer_id: int,
        device,
    ) -> torch.Tensor:
        layer_slots = self.layer_slots.get(request_key, {}).get(layer_id, None)
        if not layer_slots:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.tensor(sorted(layer_slots), dtype=torch.long, device=device)

    def build_prefix_placeholder(
        self,
        request_key: str,
        logical_len: int,
        device,
    ) -> torch.Tensor:
        placeholder = torch.zeros(logical_len, dtype=torch.int64, device=device)
        live_slots = self.get_live_slots(request_key)
        if not live_slots:
            return placeholder
        limit = min(logical_len, len(live_slots))
        placeholder[:limit] = torch.tensor(
            live_slots[:limit],
            dtype=torch.int64,
            device=device,
        )
        return placeholder

    def release_unowned_slots(self, request_key: str, slots: torch.Tensor):
        if slots is None or slots.numel() == 0:
            return
        request_ref_counts = self.slot_ref_counts.get(request_key, {})
        request_observers = self.slot_observers.get(request_key, {})
        to_free = {
            slot
            for slot in _tensor_to_slot_set(slots)
            if request_ref_counts.get(slot, 0) <= 0
            and len(request_observers.get(slot, set())) >= self.num_layers
        }
        self._free_slots(to_free)


class SemanticKVProjectionLayer(nn.Module):
    """Replicated low-rank projector used for semantic-KV experiments.

    The weight is intentionally replicated across tensor-parallel ranks because it is
    lightweight. Runtime request state is stored here so the torch_native attention
    backend can apply hard semantic cache compression during rollout.
    """

    def __init__(
        self,
        low_rank_dim: int,
        head_dim: int,
        budget_ratio: float,
        sink_window_size: int,
        recent_window_size: int,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.low_rank_dim = low_rank_dim
        self.head_dim = head_dim
        self.budget_ratio = budget_ratio
        self.sink_window_size = sink_window_size
        self.recent_window_size = recent_window_size
        self.weight = Parameter(torch.empty(low_rank_dim, head_dim, dtype=params_dtype))
        _init_orthogonal_param(self.weight)
        self.request_states: Dict[str, Dict[str, Union[torch.Tensor, int]]] = {}
        self.compaction_manager: Optional[SemanticKVCompactionManager] = None

    def cache_budget(self, seq_len: int) -> int:
        return min(
            int(seq_len * self.budget_ratio)
            + self.sink_window_size
            + self.recent_window_size,
            seq_len,
        )

    def project_token_keys(self, key_states: torch.Tensor) -> torch.Tensor:
        token_keys = key_states.mean(dim=1).float()
        return F.linear(token_keys, self.weight.float())

    def get_request_state(self, request_key: str):
        return self.request_states.get(request_key, None)

    def set_request_state(
        self,
        request_key: str,
        state: Dict[str, Union[torch.Tensor, int]],
    ):
        self.request_states[request_key] = state

    def reset_request_state(self, request_key: str):
        self.request_states.pop(request_key, None)

    def extra_repr(self) -> str:
        return (
            f"low_rank_dim={self.low_rank_dim}, head_dim={self.head_dim}, "
            f"budget_ratio={self.budget_ratio}, "
            f"sink_window_size={self.sink_window_size}, "
            f"recent_window_size={self.recent_window_size}"
        )


class LearnedLokiProjectionLayer(nn.Module):
    """Replicated low-rank scorer used for Learned-Loki rollout."""

    def __init__(
        self,
        low_rank_dim: int,
        head_dim: int,
        budget_ratio: float,
        sink_window_size: int,
        recent_window_size: int,
        gate_temperature: float,
        threshold_init: float,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.low_rank_dim = low_rank_dim
        self.head_dim = head_dim
        self.budget_ratio = budget_ratio
        self.sink_window_size = sink_window_size
        self.recent_window_size = recent_window_size
        self.weight = Parameter(torch.empty(low_rank_dim, head_dim, dtype=params_dtype))
        self.threshold = Parameter(
            torch.full((), threshold_init, dtype=params_dtype or torch.float32)
        )
        self.gate_temperature_param = Parameter(
            torch.full((), gate_temperature, dtype=params_dtype or torch.float32),
            requires_grad=False,
        )
        _init_orthogonal_param(self.weight)

    def cache_budget(self, seq_len: int) -> int:
        protected = self.sink_window_size + self.recent_window_size
        middle_tokens = max(seq_len - protected, 0)
        middle_budget = min(int(middle_tokens * self.budget_ratio), middle_tokens)
        return min(protected + middle_budget, seq_len)

    def project_token_keys(self, key_states: torch.Tensor) -> torch.Tensor:
        token_keys = key_states.float()
        return F.linear(token_keys, self.weight.float())

    @property
    def gate_temperature(self) -> float:
        return float(self.gate_temperature_param.detach().item())

    @gate_temperature.setter
    def gate_temperature(self, value: float):
        with torch.no_grad():
            self.gate_temperature_param.fill_(float(value))

    def extra_repr(self) -> str:
        return (
            f"low_rank_dim={self.low_rank_dim}, head_dim={self.head_dim}, "
            f"budget_ratio={self.budget_ratio}, "
            f"sink_window_size={self.sink_window_size}, "
            f"recent_window_size={self.recent_window_size}, "
            f"gate_temperature={self.gate_temperature}"
        )


def monkey_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    attn_kwargs = {"save_kv_cache": True}
    if hasattr(self, "adapter"):
        attn_kwargs["adapter"] = self.adapter
    if hasattr(self, "semantic_kv"):
        attn_kwargs["semantic_kv"] = self.semantic_kv
    if hasattr(self, "learned_loki"):
        attn_kwargs["learned_loki"] = self.learned_loki
    attn_output = self.attn(
        q,
        k,
        v,
        forward_batch,
        **attn_kwargs,
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
    attn_kwargs = {"save_kv_cache": True}
    if hasattr(self, "adapter"):
        attn_kwargs["adapter"] = self.adapter
    if hasattr(self, "semantic_kv"):
        attn_kwargs["semantic_kv"] = self.semantic_kv
    if hasattr(self, "learned_loki"):
        attn_kwargs["learned_loki"] = self.learned_loki
    attn_output = self.attn(
        q,
        k,
        v,
        forward_batch,
        **attn_kwargs,
    )
    output, _ = self.o_proj(attn_output)
    return output
