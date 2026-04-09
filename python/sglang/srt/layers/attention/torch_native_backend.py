from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Union

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.monkey_forward import (
        LearnedLokiProjectionLayer,
        SemanticKVProjectionLayer,
    )


def _repeat_kv_tokens(
    tokens: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    if num_key_value_groups == 1:
        return tokens
    return tokens.repeat_interleave(num_key_value_groups, dim=1)


def _semantic_attention_step(
    query_state: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    num_key_value_groups: int,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded_keys = _repeat_kv_tokens(key_states, num_key_value_groups).transpose(0, 1)
    expanded_values = _repeat_kv_tokens(value_states, num_key_value_groups).transpose(
        0, 1
    )
    attn_logits = (
        torch.einsum("hd,hsd->hs", query_state.float(), expanded_keys.float())
        * scaling
    )
    attn_probs = torch.softmax(attn_logits, dim=-1, dtype=torch.float32)
    output = torch.einsum(
        "hs,hsd->hd",
        attn_probs.to(expanded_values.dtype),
        expanded_values,
    )
    return output.to(query_state.dtype), attn_probs


def _split_windows(
    total_tokens: int,
    sink_window_size: int,
    recent_window_size: int,
) -> tuple[int, int]:
    keep_sink = min(sink_window_size, total_tokens)
    keep_recent = min(recent_window_size, max(total_tokens - keep_sink, 0))
    middle_end = total_tokens - keep_recent
    return keep_sink, middle_end


def _compute_learned_loki_gates(
    query_state: torch.Tensor,
    key_states: torch.Tensor,
    learned_loki: "LearnedLokiProjectionLayer",
    num_key_value_groups: int,
) -> torch.Tensor:
    total_tokens = key_states.shape[0]
    gates = key_states.new_ones((total_tokens,), dtype=torch.float32)
    keep_sink, middle_end = _split_windows(
        total_tokens,
        sink_window_size=learned_loki.sink_window_size,
        recent_window_size=learned_loki.recent_window_size,
    )
    if middle_end <= keep_sink:
        return gates

    projected_query = torch.nn.functional.linear(
        query_state.float(),
        learned_loki.weight.float(),
    )
    projected_keys = learned_loki.project_token_keys(key_states)
    expanded_projected_keys = _repeat_kv_tokens(
        projected_keys,
        num_key_value_groups,
    ).transpose(0, 1)
    approx_scores = (
        torch.einsum(
            "hr,hsr->hs",
            projected_query.float(),
            expanded_projected_keys.float(),
        )
        / math.sqrt(learned_loki.low_rank_dim)
    ).mean(dim=0)
    gates[keep_sink:middle_end] = torch.sigmoid(
        (
            approx_scores[keep_sink:middle_end]
            - learned_loki.threshold.float()
        )
        / max(learned_loki.gate_temperature, 1e-6)
    )
    return gates


def _learned_loki_attention_step(
    query_state: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    gates: torch.Tensor,
    num_key_value_groups: int,
    scaling: float,
) -> torch.Tensor:
    expanded_keys = _repeat_kv_tokens(key_states, num_key_value_groups).transpose(0, 1)
    expanded_values = _repeat_kv_tokens(value_states, num_key_value_groups).transpose(
        0,
        1,
    )
    attn_logits = (
        torch.einsum("hd,hsd->hs", query_state.float(), expanded_keys.float())
        * scaling
    )
    attn_logits = attn_logits + torch.log(gates.clamp_min(1e-6)).unsqueeze(0)
    attn_probs = torch.softmax(attn_logits, dim=-1, dtype=torch.float32)
    output = torch.einsum(
        "hs,hsd->hd",
        attn_probs.to(expanded_values.dtype),
        expanded_values,
    )
    return output.to(query_state.dtype)


def _hard_greedy_select(
    candidate_features: torch.Tensor,
    candidate_scores: torch.Tensor,
    num_select: int,
    retained_features: Optional[torch.Tensor],
) -> torch.Tensor:
    if num_select <= 0 or candidate_features.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=candidate_features.device)

    current_retained = (
        retained_features
        if retained_features is not None
        else candidate_features.new_empty((0, candidate_features.shape[-1]))
    )
    remaining_mask = torch.ones(
        candidate_features.shape[0], dtype=torch.bool, device=candidate_features.device
    )
    chosen = []

    target_count = min(num_select, candidate_features.shape[0])
    while len(chosen) < target_count:
        remaining_idx = remaining_mask.nonzero(as_tuple=False).flatten()
        if remaining_idx.numel() == 0:
            break

        remaining_feats = candidate_features[remaining_idx]
        remaining_scores = candidate_scores[remaining_idx]
        if current_retained.numel() == 0:
            logits = torch.log(remaining_scores.clamp_min(1e-6))
        else:
            min_distance = torch.cdist(
                remaining_feats.float(),
                current_retained.float(),
                p=2,
            ).min(dim=-1).values
            logits = torch.log(remaining_scores.clamp_min(1e-6)) + torch.log(
                min_distance.clamp_min(1e-6)
            )
        best_idx = remaining_idx[logits.argmax()]
        chosen.append(best_idx.item())
        remaining_mask[best_idx] = False
        current_retained = torch.cat(
            [current_retained, candidate_features[best_idx : best_idx + 1]],
            dim=0,
        )

    return torch.tensor(chosen, dtype=torch.long, device=candidate_features.device)


def _compress_request_state(
    semantic_kv: "SemanticKVProjectionLayer",
    kv_indices: torch.Tensor,
    importance_scores: torch.Tensor,
    k_cache: torch.Tensor,
    logical_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_budget = semantic_kv.cache_budget(logical_len)
    if kv_indices.numel() <= cache_budget:
        return kv_indices, importance_scores

    keep_sink = min(semantic_kv.sink_window_size, cache_budget, kv_indices.numel())
    remaining_after_sink = max(cache_budget - keep_sink, 0)
    keep_recent = min(
        semantic_kv.recent_window_size,
        remaining_after_sink,
        kv_indices.numel() - keep_sink,
    )
    middle_end = kv_indices.numel() - keep_recent
    candidate_indices = torch.arange(
        keep_sink,
        middle_end,
        device=kv_indices.device,
        dtype=torch.long,
    )
    num_select = max(cache_budget - keep_sink - keep_recent, 0)

    live_keys = k_cache[kv_indices.long()]
    projected_keys = semantic_kv.project_token_keys(live_keys)

    protected_parts = []
    if keep_sink > 0:
        protected_parts.append(
            torch.arange(keep_sink, device=kv_indices.device, dtype=torch.long)
        )
    if keep_recent > 0:
        protected_parts.append(
            torch.arange(
                middle_end,
                kv_indices.numel(),
                device=kv_indices.device,
                dtype=torch.long,
            )
        )
    protected_indices = (
        torch.cat(protected_parts, dim=0)
        if protected_parts
        else torch.empty(0, device=kv_indices.device, dtype=torch.long)
    )
    protected_features = (
        projected_keys[protected_indices] if protected_indices.numel() > 0 else None
    )
    selected_middle = _hard_greedy_select(
        candidate_features=projected_keys[candidate_indices],
        candidate_scores=importance_scores[candidate_indices],
        num_select=num_select,
        retained_features=protected_features,
    )
    if selected_middle.numel() > 0:
        selected_middle = candidate_indices[selected_middle]

    keep_idx = torch.cat([protected_indices, selected_middle], dim=0).sort().values
    return kv_indices[keep_idx], importance_scores[keep_idx]


def _bootstrap_request_state(
    semantic_kv: "SemanticKVProjectionLayer",
    request_key: str,
    layer_id: int,
    prefix_kv_indices: torch.Tensor,
    k_cache: torch.Tensor,
    logical_len: int,
) -> Dict[str, Union[torch.Tensor, int]]:
    compaction_manager = semantic_kv.compaction_manager
    if compaction_manager is not None:
        kv_indices = compaction_manager.get_layer_slots_tensor(
            request_key,
            layer_id,
            k_cache.device,
        )
    else:
        kv_indices = torch.empty(0, dtype=torch.long, device=k_cache.device)

    if kv_indices.numel() == 0:
        kv_indices = prefix_kv_indices[prefix_kv_indices > 0].long().clone()

    importance_scores = torch.ones(
        kv_indices.shape[0],
        dtype=torch.float32,
        device=k_cache.device,
    )
    if kv_indices.numel() > 0:
        kv_indices, importance_scores = _compress_request_state(
            semantic_kv=semantic_kv,
            kv_indices=kv_indices,
            importance_scores=importance_scores,
            k_cache=k_cache,
            logical_len=logical_len,
        )
    return {
        "kv_indices": kv_indices,
        "importance_scores": importance_scores,
        "logical_len": logical_len,
    }


def _resolve_request_key(req_to_token_pool, req_pool_idx: int) -> str:
    request_key = req_to_token_pool.resolve_request_key(req_pool_idx)
    if request_key is not None:
        return request_key
    return str(req_pool_idx)


class TorchNativeAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out_redudant = (
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    def _run_semantic_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token_pool,
        req_pool_indices: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        cache_loc: torch.Tensor,
        layer: "RadixAttention",
        semantic_kv: "SemanticKVProjectionLayer",
    ):
        num_key_value_groups = layer.tp_q_head_num // layer.tp_k_head_num
        start_q = 0
        for seq_idx in range(req_pool_indices.shape[0]):
            req_pool_idx = int(req_pool_indices[seq_idx].item())
            request_key = _resolve_request_key(req_to_token_pool, req_pool_idx)
            prefix_len = int(extend_prefix_lens[seq_idx].item())
            extend_len = int(extend_seq_lens[seq_idx].item())
            end_q = start_q + extend_len

            if prefix_len == 0:
                semantic_kv.reset_request_state(request_key)
                if semantic_kv.compaction_manager is not None:
                    semantic_kv.compaction_manager.reset_request(request_key)
                state = {
                    "kv_indices": cache_loc.new_empty((0,), dtype=torch.long),
                    "importance_scores": k_cache.new_empty((0,), dtype=torch.float32),
                    "logical_len": 0,
                }
            else:
                state = semantic_kv.get_request_state(request_key)
                if state is None or int(state["logical_len"]) != prefix_len:
                    prefix_indices = req_to_token_pool.req_to_token[
                        req_pool_idx, :prefix_len
                    ]
                    state = _bootstrap_request_state(
                        semantic_kv=semantic_kv,
                        request_key=request_key,
                        layer_id=layer.layer_id,
                        prefix_kv_indices=prefix_indices,
                        k_cache=k_cache,
                        logical_len=prefix_len,
                    )

            kv_indices = state["kv_indices"]
            importance_scores = state["importance_scores"]
            logical_len = int(state["logical_len"])
            per_req_query = query[start_q:end_q]
            per_req_cache_loc = cache_loc[start_q:end_q]

            for token_offset in range(extend_len):
                kv_indices = torch.cat(
                    [kv_indices, per_req_cache_loc[token_offset : token_offset + 1].long()],
                    dim=0,
                )
                importance_scores = torch.cat(
                    [
                        importance_scores,
                        importance_scores.new_zeros((1,), dtype=torch.float32),
                    ],
                    dim=0,
                )
                logical_len += 1

                live_keys = k_cache[kv_indices.long()]
                live_values = v_cache[kv_indices.long()]
                token_output, attn_probs = _semantic_attention_step(
                    query_state=per_req_query[token_offset],
                    key_states=live_keys,
                    value_states=live_values,
                    num_key_value_groups=num_key_value_groups,
                    scaling=layer.scaling,
                )
                output[start_q + token_offset] = token_output
                importance_scores = importance_scores + attn_probs.sum(dim=0).to(
                    importance_scores.dtype
                )
                kv_indices, importance_scores = _compress_request_state(
                    semantic_kv=semantic_kv,
                    kv_indices=kv_indices,
                    importance_scores=importance_scores,
                    k_cache=k_cache,
                    logical_len=logical_len,
                )
                if semantic_kv.compaction_manager is not None:
                    semantic_kv.compaction_manager.update_layer_slots(
                        request_key=request_key,
                        layer_id=layer.layer_id,
                        new_slots=kv_indices,
                        observed_slots=per_req_cache_loc[token_offset : token_offset + 1],
                    )

            semantic_kv.set_request_state(
                request_key,
                {
                    "kv_indices": kv_indices,
                    "importance_scores": importance_scores,
                    "logical_len": logical_len,
                },
            )
            start_q = end_q
        return output

    def _run_semantic_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token_pool,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        cache_loc: torch.Tensor,
        layer: "RadixAttention",
        semantic_kv: "SemanticKVProjectionLayer",
    ):
        num_key_value_groups = layer.tp_q_head_num // layer.tp_k_head_num
        for seq_idx in range(req_pool_indices.shape[0]):
            req_pool_idx = int(req_pool_indices[seq_idx].item())
            request_key = _resolve_request_key(req_to_token_pool, req_pool_idx)
            seq_len = int(seq_lens[seq_idx].item())

            state = semantic_kv.get_request_state(request_key)
            expected_prefix_len = max(seq_len - 1, 0)
            if state is None or int(state["logical_len"]) != expected_prefix_len:
                prefix_indices = req_to_token_pool.req_to_token[
                    req_pool_idx, :expected_prefix_len
                ]
                state = _bootstrap_request_state(
                    semantic_kv=semantic_kv,
                    request_key=request_key,
                    layer_id=layer.layer_id,
                    prefix_kv_indices=prefix_indices,
                    k_cache=k_cache,
                    logical_len=expected_prefix_len,
                )

            kv_indices = state["kv_indices"]
            importance_scores = state["importance_scores"]
            logical_len = int(state["logical_len"])

            kv_indices = torch.cat([kv_indices, cache_loc[seq_idx : seq_idx + 1].long()], dim=0)
            importance_scores = torch.cat(
                [importance_scores, importance_scores.new_zeros((1,), dtype=torch.float32)],
                dim=0,
            )
            logical_len += 1

            live_keys = k_cache[kv_indices.long()]
            live_values = v_cache[kv_indices.long()]
            token_output, attn_probs = _semantic_attention_step(
                query_state=query[seq_idx],
                key_states=live_keys,
                value_states=live_values,
                num_key_value_groups=num_key_value_groups,
                scaling=layer.scaling,
            )
            output[seq_idx] = token_output
            importance_scores = importance_scores + attn_probs.sum(dim=0).to(
                importance_scores.dtype
            )
            kv_indices, importance_scores = _compress_request_state(
                semantic_kv=semantic_kv,
                kv_indices=kv_indices,
                importance_scores=importance_scores,
                k_cache=k_cache,
                logical_len=logical_len,
            )
            if semantic_kv.compaction_manager is not None:
                semantic_kv.compaction_manager.update_layer_slots(
                    request_key=request_key,
                    layer_id=layer.layer_id,
                    new_slots=kv_indices,
                    observed_slots=cache_loc[seq_idx : seq_idx + 1],
                )
            semantic_kv.set_request_state(
                request_key,
                {
                    "kv_indices": kv_indices,
                    "importance_scores": importance_scores,
                    "logical_len": logical_len,
                },
            )
        return output

    def _run_learned_loki_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token_pool,
        req_pool_indices: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        layer: "RadixAttention",
        learned_loki: "LearnedLokiProjectionLayer",
    ):
        num_key_value_groups = layer.tp_q_head_num // layer.tp_k_head_num
        start_q = 0
        for seq_idx in range(req_pool_indices.shape[0]):
            req_pool_idx = int(req_pool_indices[seq_idx].item())
            prefix_len = int(extend_prefix_lens[seq_idx].item())
            extend_len = int(extend_seq_lens[seq_idx].item())
            end_q = start_q + extend_len
            if extend_len <= 0:
                start_q = end_q
                continue

            full_slots = req_to_token_pool.req_to_token[
                req_pool_idx, : prefix_len + extend_len
            ].long()
            full_keys = k_cache[full_slots]
            full_values = v_cache[full_slots]
            per_req_query = query[start_q:end_q]

            for token_offset in range(extend_len):
                live_len = prefix_len + token_offset + 1
                live_keys = full_keys[:live_len]
                live_values = full_values[:live_len]
                gates = _compute_learned_loki_gates(
                    query_state=per_req_query[token_offset],
                    key_states=live_keys,
                    learned_loki=learned_loki,
                    num_key_value_groups=num_key_value_groups,
                )
                output[start_q + token_offset] = _learned_loki_attention_step(
                    query_state=per_req_query[token_offset],
                    key_states=live_keys,
                    value_states=live_values,
                    gates=gates,
                    num_key_value_groups=num_key_value_groups,
                    scaling=layer.scaling,
                )
            start_q = end_q
        return output

    def _run_learned_loki_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token_pool,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        layer: "RadixAttention",
        learned_loki: "LearnedLokiProjectionLayer",
    ):
        num_key_value_groups = layer.tp_q_head_num // layer.tp_k_head_num
        for seq_idx in range(req_pool_indices.shape[0]):
            req_pool_idx = int(req_pool_indices[seq_idx].item())
            seq_len = int(seq_lens[seq_idx].item())
            live_slots = req_to_token_pool.req_to_token[req_pool_idx, :seq_len].long()
            live_keys = k_cache[live_slots]
            live_values = v_cache[live_slots]
            gates = _compute_learned_loki_gates(
                query_state=query[seq_idx],
                key_states=live_keys,
                learned_loki=learned_loki,
                num_key_value_groups=num_key_value_groups,
            )
            output[seq_idx] = _learned_loki_attention_step(
                query_state=query[seq_idx],
                key_states=live_keys,
                value_states=live_values,
                gates=gates,
                num_key_value_groups=num_key_value_groups,
                scaling=layer.scaling,
            )
        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        semantic_kv: Optional["SemanticKVProjectionLayer"] = None,
        learned_loki: Optional["LearnedLokiProjectionLayer"] = None,
        **kwargs,
    ):
        del kwargs
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        if semantic_kv is not None:
            self._run_semantic_forward_extend(
                q_,
                o_,
                k_cache,
                v_cache,
                forward_batch.req_to_token_pool,
                forward_batch.req_pool_indices,
                forward_batch.extend_prefix_lens,
                forward_batch.extend_seq_lens,
                cache_loc,
                layer,
                semantic_kv,
            )
            return o
        if learned_loki is not None:
            self._run_learned_loki_forward_extend(
                q_,
                o_,
                k_cache,
                v_cache,
                forward_batch.req_to_token_pool,
                forward_batch.req_pool_indices,
                forward_batch.extend_prefix_lens,
                forward_batch.extend_seq_lens,
                layer,
                learned_loki,
            )
            return o

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        self._run_sdpa_forward_extend(
            q_,
            o_,
            k_cache,
            v_cache,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=causal,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        semantic_kv: Optional["SemanticKVProjectionLayer"] = None,
        learned_loki: Optional["LearnedLokiProjectionLayer"] = None,
        **kwargs,
    ):
        del kwargs
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        if semantic_kv is not None:
            self._run_semantic_forward_decode(
                q_,
                o_,
                k_cache,
                v_cache,
                forward_batch.req_to_token_pool,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                cache_loc,
                layer,
                semantic_kv,
            )
            return o
        if learned_loki is not None:
            self._run_learned_loki_forward_decode(
                q_,
                o_,
                k_cache,
                v_cache,
                forward_batch.req_to_token_pool,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                layer,
                learned_loki,
            )
            return o

        self._run_sdpa_forward_decode(
            q_,
            o_,
            k_cache,
            v_cache,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,
        )

        return o

    def support_triton(self):
        return False
