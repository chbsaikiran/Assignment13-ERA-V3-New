import torch
import torch.nn as nn
import math

# RMSNorm is a normalization technique that normalizes the input by dividing by the square root of the variance plus a small number to prevent division by zero
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5): # the number of features/dimensions/embeddings in the input, eps is a small number to prevent division by zero
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) # weight is a learnable parameter that scales the input
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).sqrt() + self.eps # compute the norm of the input
        return x / norm * self.weight # normalize the input by dividing by the norm and scale it by the weight parameter


# RotaryEmbedding is a technique that rotates the input by a learnable angle
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, device=None): # dim is the number of features/dimensions/embeddings in the input, base is a base number for the frequency, device is the device to store the buffer
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim)) # compute the inverse frequency
        self.register_buffer("inv_freq", inv_freq) # register the inverse frequency as a buffer

    def forward(self, x, seq_len):
        seq_len = seq_len.to(x.device) # convert seq_len to the device of the input 
        t = torch.arange(seq_len, device=x.device) # create a tensor of the sequence length
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # compute the frequency by taking the dot product of the sequence length and the inverse frequency
        emb = torch.cat((freqs, freqs), dim=-1) # concatenate the frequency with itself
        return emb

class LlamaMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False) # create the gate projection layer with the input dimension and the hidden dimension
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False) # create the up projection layer with the input dimension and the hidden dimension
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False) # create the down projection layer with the hidden dimension and the output dimension
        self.act_fn = nn.SiLU() # create the activation function

    def forward(self, x):
        gated = self.gate_proj(x) # apply the gate projection to the input
        hidden = self.up_proj(x) # apply the up projection to the input
        return self.down_proj(self.act_fn(gated * hidden)) # apply the activation function to the gated and hidden values and then apply the down projection
    
class LlamaAttention(nn.Module):
    def __init__(self, dim, num_heads=8,max_seq_len=2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        batch_size, seq_len, dim = x.size() # [batch_size, seq_len, dim] -> [4, 128, 576]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)


        # Split heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [batch_size, num_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)

        # Combine heads
        context = context.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.o_proj(context)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads,max_position_embeddings):
        super().__init__()
        self.self_attn = LlamaAttention(dim, num_heads,max_position_embeddings)
        self.mlp = LlamaMLP(dim, hidden_dim)
        self.input_layernorm = LlamaRMSNorm(dim)
        self.post_attention_layernorm = LlamaRMSNorm(dim)

    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual
        return x


class LlamaModel(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, hidden_dim, num_heads,max_position_embeddings):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(dim, hidden_dim, num_heads,max_position_embeddings) for _ in range(num_layers)
        ])
        self.norm = LlamaRMSNorm(dim)
        self.rotary_emb = LlamaRotaryEmbedding(dim)
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.wte = nn.Embedding(self.vocab_size, self.dim)
        self.wpe = nn.Embedding(self.max_position_embeddings, self.dim)

    def forward(self, tokens):
        B, T = tokens.size()
        assert T <= self.max_position_embeddings, f"Cannot forward sequence of length {T}, block size is only {self.max_position_embeddings}"

        pos = torch.arange(0, T, dtype=torch.long, device=tokens.device) # shape (T)
        pos_emb = self.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.wte(tokens) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab_size = config.vocab_size
        dim = config.hidden_size
        num_layers = config.num_layers
        hidden_dim = config.intermediate_size
        num_heads = config.num_attention_heads
        max_position_embeddings = config.max_position_embeddings

        self.model = LlamaModel(vocab_size, dim, num_layers, hidden_dim, num_heads,max_position_embeddings)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.model(x)
        return self.lm_head(x)