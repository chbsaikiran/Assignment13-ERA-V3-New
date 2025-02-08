# model.py

import torch
import torch.nn as nn


class PipelineBlock(nn.Module):
    """
    Wraps a block (nn.Module) with an associated pipeline rank.
    If no block is provided, the block acts as an identity.
    """
    def __init__(self, pp_rank: int, pp_block: nn.Module = None):
        super().__init__()
        self.pp_rank = pp_rank
        self.pp_block = pp_block

    def forward(self, x):
        if self.pp_block is not None:
            return self.pp_block(x)
        return x


class TensorParallelEmbedding(nn.Module):
    """
    A dummy tensor-parallel embedding.
    """
    def __init__(self, tp_rank: int, num_embeddings: int, embedding_dim: int, unsharded_num_embeddings: int):
        super().__init__()
        self.tp_rank = tp_rank
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


class Embedding(nn.Module):
    """
    A simple wrapper around a token embedding.
    """
    def __init__(self, token_embedding: nn.Module):
        super().__init__()
        self.token_embedding = token_embedding

    def forward(self, x):
        return self.token_embedding(x)


class TritonRMSNorm(nn.Module):
    """
    A dummy RMS normalization layer.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # Compute RMS norm along the last dimension.
        norm = x.norm(2, dim=-1, keepdim=True)
        return x / (norm + self.eps)


class TensorParallelColumnLinear(nn.Module):
    """
    A dummy tensor-parallel column linear layer.
    """
    def __init__(self, tp_rank: int, in_features: int, out_features: int, bias: bool = True, unsharded_out_features: int = None):
        super().__init__()
        self.tp_rank = tp_rank
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)


class TensorParallelRowLinear(nn.Module):
    """
    A dummy tensor-parallel row linear layer.
    """
    def __init__(self, tp_rank: int, in_features: int, out_features: int, bias: bool = True, unsharded_in_features: int = None):
        super().__init__()
        self.tp_rank = tp_rank
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)


class LlamaRotaryEmbedding(nn.Module):
    """
    Dummy Llama rotary embedding.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # In a real implementation, apply rotary embeddings.
        return x


class RotaryEmbedding(nn.Module):
    """
    Dummy rotary embedding.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # In a real implementation, apply rotary embeddings.
        return x


class CoreAttention(nn.Module):
    """
    Dummy attention computation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Dummy attention: simply return the input.
        return x


class CausalSelfAttention(nn.Module):
    """
    Implements a causal self-attention block.
    """
    def __init__(self, qkv_proj: nn.Module, rotary_embedding: nn.Module,
                 flash_rotary_embedding: nn.Module, o_proj: nn.Module, attention: nn.Module):
        super().__init__()
        self.qkv_proj = qkv_proj
        self.rotary_embedding = rotary_embedding
        self.flash_rotary_embedding = flash_rotary_embedding
        self.o_proj = o_proj
        self.attention = attention

    def forward(self, x):
        # Project input to QKV space.
        qkv = self.qkv_proj(x)
        # (Optionally, apply rotary embeddings here)
        # Compute attention (dummy implementation).
        attn_out = self.attention(qkv)
        # Project back.
        out = self.o_proj(attn_out)
        return out


class SiLUActivation(nn.Module):
    """
    A simple SiLU activation.
    """
    def __init__(self):
        super().__init__()
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(x)


class GLUActivation(nn.Module):
    """
    A dummy GLU activation which wraps another activation.
    """
    def __init__(self, act: nn.Module):
        super().__init__()
        self.act = act

    def forward(self, x):
        # In a full implementation, you might split the input and apply gating.
        # Here we simply apply the activation.
        return self.act(x)


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) block.
    """
    def __init__(self, gate_up_proj: nn.Module, down_proj: nn.Module, split_silu_mul: nn.Module):
        super().__init__()
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.split_silu_mul = split_silu_mul

    def forward(self, x):
        # Project up.
        gate = self.gate_up_proj(x)
        # Apply activation.
        activated = self.split_silu_mul(gate)
        # Project down.
        out = self.down_proj(activated)
        return out


class LlamaDecoderLayer(nn.Module):
    """
    One decoder layer in the Llama model.
    """
    def __init__(self, input_layernorm: nn.Module, attn: nn.Module,
                 post_attention_layernorm: nn.Module, mlp: nn.Module):
        super().__init__()
        self.input_layernorm = input_layernorm
        self.attn = attn
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp

    def forward(self, x):
        # First sub-layer: attention.
        residual = x
        x_norm = self.input_layernorm(x)
        attn_out = self.attn(x_norm)
        x = residual + attn_out

        # Second sub-layer: MLP.
        residual = x
        x_norm = self.post_attention_layernorm(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out
        return x


class LlamaModel(nn.Module):
    """
    The core Llama model used for training.
    """
    def __init__(self, token_position_embeddings: nn.Module, decoder: nn.Module,
                 final_layer_norm: nn.Module, lm_head: nn.Module, cast_to_fp32: nn.Module):
        super().__init__()
        self.token_position_embeddings = token_position_embeddings
        self.decoder = decoder  # Typically a nn.ModuleList of decoder layers.
        self.final_layer_norm = final_layer_norm
        self.lm_head = lm_head
        self.cast_to_fp32 = cast_to_fp32

    def forward(self, input_ids):
        # Embed the input tokens.
        x = self.token_position_embeddings(input_ids)
        # Pass through each decoder layer.
        for layer in self.decoder:
            x = layer(x)
        # Final normalization.
        x = self.final_layer_norm(x)
        # Project to vocabulary logits.
        x = self.lm_head(x)
        # Optionally cast to fp32.
        x = self.cast_to_fp32(x)
        return x


class Loss(nn.Module):
    """
    Loss module using cross-entropy.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # Reshape logits and targets as required by CrossEntropyLoss.
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        return self.loss_fn(logits, targets)


class LlamaForTraining(nn.Module):
    """
    Wraps the Llama model and loss for training.
    """
    def __init__(self, model: nn.Module, loss: nn.Module):
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, input_ids, targets=None):
        logits = self.model(input_ids)
        if targets is not None:
            loss_value = self.loss(logits, targets)
            return logits, loss_value
        return logits


def build_llama_for_training() -> LlamaForTraining:
    """Constructs the LlamaForTraining model with the architecture provided."""
    # Token position embeddings
    token_embedding = TensorParallelEmbedding(
        tp_rank=0,
        num_embeddings=49152,
        embedding_dim=576,
        unsharded_num_embeddings=49152
    )
    embedding_block = Embedding(token_embedding=token_embedding)
    token_position_embeddings = PipelineBlock(pp_rank=0, pp_block=embedding_block)

    # Build decoder layers (30 layers)
    decoder_layers = []
    for _ in range(30):
        # Create the components for one decoder layer
        input_layernorm = TritonRMSNorm()

        qkv_proj = TensorParallelColumnLinear(
            tp_rank=0,
            in_features=576,
            out_features=960,
            bias=False,
            unsharded_out_features=960
        )
        rotary_embedding = LlamaRotaryEmbedding()
        flash_rotary_embedding = RotaryEmbedding()
        o_proj = TensorParallelRowLinear(
            tp_rank=0,
            in_features=576,
            out_features=576,
            bias=False,
            unsharded_in_features=576
        )
        attention = CoreAttention()
        attn = CausalSelfAttention(
            qkv_proj=qkv_proj,
            rotary_embedding=rotary_embedding,
            flash_rotary_embedding=flash_rotary_embedding,
            o_proj=o_proj,
            attention=attention
        )

        post_attention_layernorm = TritonRMSNorm()

        gate_up_proj = TensorParallelColumnLinear(
            tp_rank=0,
            in_features=576,
            out_features=3072,
            bias=False,
            unsharded_out_features=3072
        )
        down_proj = TensorParallelRowLinear(
            tp_rank=0,
            in_features=1536,
            out_features=576,
            bias=False,
            unsharded_in_features=1536
        )
        silu_activation = SiLUActivation()
        split_silu_mul = GLUActivation(act=silu_activation)
        mlp = MLP(
            gate_up_proj=gate_up_proj,
            down_proj=down_proj,
            split_silu_mul=split_silu_mul
        )

        decoder_layer = LlamaDecoderLayer(
            input_layernorm=input_layernorm,
            attn=attn,
            post_attention_layernorm=post_attention_layernorm,
            mlp=mlp
        )

        # Wrap the decoder layer into a PipelineBlock.
        pipeline_decoder_layer = PipelineBlock(pp_rank=0, pp_block=decoder_layer)
        decoder_layers.append(pipeline_decoder_layer)

    # Wrap all decoder layers in a ModuleList.
    decoder = nn.ModuleList(decoder_layers)

    # Final layer normalization.
    final_layer_norm_block = TritonRMSNorm()
    final_layer_norm = PipelineBlock(pp_rank=0, pp_block=final_layer_norm_block)

    # LM head for projecting to the vocabulary.
    lm_head_linear = TensorParallelColumnLinear(
        tp_rank=0,
        in_features=576,
        out_features=49152,
        bias=False,
        unsharded_out_features=49152
    )
    lm_head = PipelineBlock(pp_rank=0, pp_block=lm_head_linear)

    # A pipeline block that casts outputs to fp32 (if needed).
    cast_to_fp32 = PipelineBlock(pp_rank=0)

    # Build the Llama model.
    llama_model = LlamaModel(
        token_position_embeddings=token_position_embeddings,
        decoder=decoder,
        final_layer_norm=final_layer_norm,
        lm_head=lm_head,
        cast_to_fp32=cast_to_fp32
    )

    # Loss wrapped in a pipeline block.
    loss_pipeline = PipelineBlock(pp_rank=0, pp_block=Loss())

    # Wrap the model and loss together.
    llama_for_training = LlamaForTraining(model=llama_model, loss=loss_pipeline)

    return llama_for_training


#if __name__ == "__main__":
#    # Test the model construction with dummy inputs.
#    model = build_llama_for_training()
#    print("LlamaForTraining model:")
#    print(model)
#
#    # Create dummy input data: e.g., batch_size=1, sequence_length=10.
#    dummy_input = torch.randint(0, 49152, (1, 10))
#    logits = model(dummy_input)
#    print("Logits shape:", logits.shape)
