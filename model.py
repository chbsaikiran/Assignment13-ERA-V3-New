# coding=utf-8
# Copyright 2018 HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch LLaMa model."""

from typing import Dict, List, Optional, Union

import torch
from torch import nn
from torch.utils.checkpoint import CheckpointFunction


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, end: int, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.end = end
        self.theta = theta
        self.register_buffer("freqs_cis", torch.empty(0), persistent=False)
        self._initialized_buffer = False

    def init_rotary_embeddings(self):
        if self._initialized_buffer is True:
            return
        self.freqs_cis = torch.empty(self.end, self.dim // 2, 2, dtype=torch.float, device="cuda")
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device="cpu")[: (self.dim // 2)] / self.dim)
        ).to("cuda")
        t = torch.arange(self.end, device="cuda")
        freqs = torch.outer(t, freqs).float()
        complex_freqs = torch.polar(torch.ones_like(freqs), freqs)
        self.freqs_cis.copy_(torch.view_as_real(complex_freqs))
        self._initialized_buffer = True

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.LongTensor]):
        batch_size, seq_length, num_heads, inner_dim = x.shape
        if position_ids is not None and position_ids[-1, -1] >= self.end:
            self.end *= 2
            self._initialized_buffer = False
        if self._initialized_buffer is False:
            self.init_rotary_embeddings()
        dtype = x.dtype
        assert inner_dim % 2 == 0
        x = x.view(batch_size, seq_length, num_heads, inner_dim // 2, 2)
        complex_x = torch.view_as_complex(x)
        freqs_cis = self.freqs_cis[None, :seq_length, None, :] if position_ids is None else self.freqs_cis[position_ids][:, :, None, :]
        complex_freqs = torch.view_as_complex(freqs_cis)
        x_out = torch.view_as_real(complex_x * complex_freqs).view(batch_size, seq_length, num_heads, inner_dim)
        return x_out.type(dtype)


class GLUActivation(nn.Module):
    def __init__(self, act_fn_name: str):
        super().__init__()
        self.act = getattr(torch.nn.functional, act_fn_name)

    def forward(self, merged_states: torch.Tensor):
        gate_states, up_states = torch.split(merged_states, merged_states.shape[-1] // 2, dim=-1)
        return self.act(gate_states) * up_states


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.split_silu_mul = GLUActivation(activation)

    def forward(self, hidden_states):
        merged_states = self.gate_up_proj(hidden_states)
        hidden_states = self.down_proj(self.split_silu_mul(merged_states))
        return hidden_states


class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size, activation)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states + residual


class LlamaModel(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str, num_layers: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)  # Add embedding layer
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(hidden_size, intermediate_size, activation) for _ in range(num_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)  # Output layer mapping hidden states to vocab

    def forward(self, input_ids):
        hidden_states = self.embedding(input_ids)  # Convert token indices to embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.output_layer(hidden_states)  # Map to vocab size
        return logits