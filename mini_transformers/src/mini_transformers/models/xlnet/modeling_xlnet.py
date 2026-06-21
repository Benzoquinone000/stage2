"""XLNet with two-stream relative self-attention.

This implementation keeps the central XLNet mechanics: permutation masks,
content/query streams, relative position scores, segment-aware attention,
optional memory, target mapping, and a language-modeling head.
"""

from __future__ import annotations

from math import sqrt

import torch
from torch import nn

from mini_transformers.configs import XLNetConfig
from mini_transformers.models.modeling_utils import PreTrainedModel
from mini_transformers.modules import FeedForward

from .heads import XLNetLMHead


def build_permutation_mask(perm_order: torch.Tensor) -> torch.Tensor:
    """Return mask[i, j] = 1 when position i cannot attend to position j."""

    return (perm_order[:, None, :] > perm_order[:, :, None]).long()


def relative_position_ids(q_len: int, k_len: int, max_position: int, device: torch.device) -> torch.Tensor:
    query_pos = torch.arange(k_len - q_len, k_len, device=device)[:, None]
    key_pos = torch.arange(k_len, device=device)[None, :]
    distances = key_pos - query_pos
    distances = distances.clamp(-max_position + 1, max_position - 1)
    return distances + max_position - 1


def prepend_memory_mask(mask: torch.Tensor, mem_len: int) -> torch.Tensor:
    if mem_len == 0:
        return mask
    memory = mask.new_ones(mask.size(0), mask.size(1), mem_len)
    return torch.cat([memory, mask], dim=-1)


class XLNetRelativeAttention(nn.Module):
    def __init__(self, config: XLNetConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_position = config.max_position_embeddings

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.relative_embeddings = nn.Embedding(2 * self.max_position - 1, config.hidden_size)
        self.segment_embeddings = nn.Parameter(torch.empty(2, self.num_heads, self.head_dim))
        self.content_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.position_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.segment_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.dropout = nn.Dropout(config.dropout)
        nn.init.normal_(self.segment_embeddings, mean=0.0, std=0.02)

    def split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = tensor.shape
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, seq_len, self.num_heads * self.head_dim)

    def forward(
        self,
        query_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        query_token_type_ids: torch.Tensor | None = None,
        key_token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.split_heads(self.q_proj(query_states))
        k = self.split_heads(self.k_proj(key_value_states))
        v = self.split_heads(self.v_proj(key_value_states))

        q_len = query_states.size(1)
        k_len = key_value_states.size(1)
        rel_ids = relative_position_ids(q_len, k_len, self.max_position, query_states.device)
        rel = self.relative_embeddings(rel_ids)
        rel = rel.view(q_len, k_len, self.num_heads, self.head_dim).permute(2, 0, 1, 3)

        content_scores = torch.einsum("bnqd,bnkd->bnqk", q + self.content_bias[None, :, None], k)
        position_scores = torch.einsum("bnqd,nqkd->bnqk", q + self.position_bias[None, :, None], rel)
        scores = (content_scores + position_scores) / sqrt(self.head_dim)

        if query_token_type_ids is not None and key_token_type_ids is not None:
            segment_mat = (query_token_type_ids[:, :, None] != key_token_type_ids[:, None, :]).long()
            segment_scores = torch.einsum(
                "bnqd,bqknd->bnqk",
                q + self.segment_bias[None, :, None],
                self.segment_embeddings[segment_mat],
            )
            scores = scores + segment_scores

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[:, None] == 0, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, v)
        return self.o_proj(self.merge_heads(context)), probs


class XLNetLayer(nn.Module):
    def __init__(self, config: XLNetConfig) -> None:
        super().__init__()
        self.attention = XLNetRelativeAttention(config)
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config.hidden_size, config.intermediate_size, config.dropout)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def apply_ffn(self, states: torch.Tensor) -> torch.Tensor:
        return self.ffn_norm(states + self.ffn(states))

    def forward(
        self,
        content_states: torch.Tensor,
        query_states: torch.Tensor | None,
        key_value_states: torch.Tensor,
        content_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        query_token_type_ids: torch.Tensor | None = None,
        key_token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_output, _ = self.attention(
            content_states,
            key_value_states,
            content_mask,
            query_token_type_ids=query_token_type_ids,
            key_token_type_ids=key_token_type_ids,
        )
        content_states = self.attn_norm(content_states + self.dropout(attn_output))
        content_states = self.apply_ffn(content_states)

        if query_states is None:
            return content_states, None

        query_output, _ = self.attention(
            query_states,
            key_value_states,
            query_mask,
            query_token_type_ids=query_token_type_ids,
            key_token_type_ids=key_token_type_ids,
        )
        query_states = self.attn_norm(query_states + self.dropout(query_output))
        query_states = self.apply_ffn(query_states)
        return content_states, query_states


class XLNetModel(PreTrainedModel):
    config_class = XLNetConfig

    def __init__(self, config: XLNetConfig) -> None:
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.mask_embedding = nn.Parameter(torch.zeros(config.hidden_size))
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([XLNetLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        perm_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        target_mapping: torch.Tensor | None = None,
        mems: list[torch.Tensor] | None = None,
        use_query_stream: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        batch_size, seq_len = input_ids.shape
        mems = mems or [None] * len(self.layers)

        content_states = self.dropout(self.word_embeddings(input_ids))
        query_states = self.init_query_stream(batch_size, seq_len, input_ids.device) if use_query_stream else None

        content_mask = self.build_attention_mask(
            attention_mask,
            perm_mask,
            batch_size,
            seq_len,
            device=input_ids.device,
            block_self=False,
        )
        query_mask = self.build_attention_mask(
            attention_mask,
            perm_mask,
            batch_size,
            seq_len,
            device=input_ids.device,
            block_self=True,
        )

        hidden_states_for_memory = []
        for layer, memory in zip(self.layers, mems):
            mem_len = 0 if memory is None else memory.size(1)
            key_value_states = content_states if memory is None else torch.cat([memory, content_states], dim=1)
            layer_content_mask = prepend_memory_mask(content_mask, mem_len)
            layer_query_mask = prepend_memory_mask(query_mask, mem_len)
            key_token_type_ids = self.build_key_token_type_ids(token_type_ids, mem_len)

            hidden_states_for_memory.append(content_states)
            content_states, query_states = layer(
                content_states,
                query_states,
                key_value_states,
                content_mask=layer_content_mask,
                query_mask=layer_query_mask,
                query_token_type_ids=token_type_ids,
                key_token_type_ids=key_token_type_ids,
            )

        content_states = self.final_norm(content_states)
        if query_states is not None:
            query_states = self.final_norm(query_states)
            if target_mapping is not None:
                query_states = torch.matmul(target_mapping, query_states)

        new_mems = self.update_mems(hidden_states_for_memory)
        return {
            "last_hidden_state": query_states if use_query_stream else content_states,
            "content_stream": content_states,
            "query_stream": query_states,
            "mems": new_mems,
        }

    def init_query_stream(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        query_states = self.mask_embedding[None, None].expand(batch_size, seq_len, -1)
        return self.dropout(query_states)

    def build_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        perm_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        block_self: bool,
    ) -> torch.Tensor:
        allowed = torch.ones(batch_size, seq_len, seq_len, device=device, dtype=torch.long)

        if perm_mask is not None:
            allowed = allowed * (1 - perm_mask.long())
        if attention_mask is not None:
            allowed = allowed * attention_mask[:, None, :].long()
        if block_self:
            eye = torch.eye(seq_len, device=device, dtype=torch.long)[None]
            allowed = allowed * (1 - eye)
        return allowed

    def build_key_token_type_ids(
        self,
        token_type_ids: torch.Tensor | None,
        mem_len: int,
    ) -> torch.Tensor | None:
        if token_type_ids is None:
            return None
        if mem_len == 0:
            return token_type_ids
        memory_token_types = token_type_ids.new_zeros(token_type_ids.size(0), mem_len)
        return torch.cat([memory_token_types, token_type_ids], dim=1)

    def update_mems(self, hidden_states: list[torch.Tensor]) -> list[torch.Tensor] | None:
        if self.config.mem_len <= 0:
            return None
        return [states[:, -self.config.mem_len :].detach() for states in hidden_states]


class XLNetLMHeadModel(PreTrainedModel):
    config_class = XLNetConfig

    def __init__(self, config: XLNetConfig) -> None:
        super().__init__(config)
        self.xlnet = XLNetModel(config)
        self.lm_head = XLNetLMHead(config.hidden_size, config.vocab_size)
        self.lm_head.proj.weight = self.xlnet.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        perm_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        target_mapping: torch.Tensor | None = None,
        mems: list[torch.Tensor] | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        outputs = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            perm_mask=perm_mask,
            token_type_ids=token_type_ids,
            target_mapping=target_mapping,
            mems=mems,
            use_query_stream=True,
        )
        logits = self.lm_head(outputs["last_hidden_state"])
        result = {"logits": logits, "mems": outputs["mems"]}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return result
