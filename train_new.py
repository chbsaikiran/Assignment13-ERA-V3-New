import torch
from torch import nn


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

    def forward(self, x: torch.Tensor):
        batch_size, seq_length, num_heads, inner_dim = x.shape
        
        self.init_rotary_embeddings()
        dtype = x.dtype
        assert inner_dim % 2 == 0, f"inner_dim must be even, got {inner_dim}"
        
        # Reshape x to complex representation
        x = x.view(batch_size, seq_length, num_heads, inner_dim // 2, 2)
        complex_x = torch.view_as_complex(x)
        
        # Correctly reshape `freqs_cis` (must keep last dimension of size 2)
        freqs_cis = self.freqs_cis[:seq_length, :inner_dim // 2, :]
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2).expand(batch_size, seq_length, num_heads, inner_dim // 2, 2)
        
        # Ensure last dimension is 2 before conversion
        assert freqs_cis.shape[-1] == 2, f"freqs_cis last dimension must be 2, got {freqs_cis.shape}"
        
        # Convert to complex representation
        complex_freqs = torch.view_as_complex(freqs_cis)
        
        assert complex_x.shape == complex_freqs.shape, f"Shape mismatch: {complex_x.shape} vs {complex_freqs.shape}"
        
        # Apply rotary transformation
        x_out = torch.view_as_real(complex_x * complex_freqs).view(batch_size, seq_length, num_heads, inner_dim)
        
        return x_out.type(dtype)




class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, hidden_states, mask=None):
        batch_size, seq_length, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(batch_size, self.num_heads, seq_length, 3 * self.head_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        assert attn_scores.shape == (batch_size, self.num_heads, seq_length, seq_length), f"attn_scores shape mismatch: {attn_scores.shape}"

        
        #print(f"attn_scores shape: {attn_scores.shape}")  # Expected: (batch_size, num_heads, seq_length, seq_length)
        #print(f"mask shape: {mask.shape}")  # Expected: (batch_size, num_heads, seq_length, seq_length)

        
        if mask is not None:
            # Ensure `mask` has shape `(batch_size, num_heads, seq_length, seq_length)`
            assert mask.shape == (batch_size, self.num_heads, seq_length, seq_length), f"Mask shape mismatch: {mask.shape} vs expected ({batch_size}, {self.num_heads}, {seq_length}, {seq_length})"
            
            mask = mask.to(attn_scores.device)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        
        attn_probs = self.softmax(attn_scores)
        attn_output = torch.matmul(attn_probs, v)
        
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        return self.o_proj(attn_output)





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
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str, num_heads: int):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.rotary_embedding = RotaryEmbedding(hidden_size, 2048)
        self.attention = Attention(hidden_size, num_heads)
        self.mlp = MLP(hidden_size, intermediate_size, activation)

    def forward(self, hidden_states, mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Reshape before passing to RotaryEmbedding
        batch_size, seq_length, hidden_size = hidden_states.shape
        num_heads = self.attention.num_heads
        head_dim = hidden_size // num_heads
        
        hidden_states = hidden_states.view(batch_size, seq_length, num_heads, head_dim)
        
        hidden_states = self.rotary_embedding(hidden_states)
        
        # Reshape back
        hidden_states = hidden_states.view(batch_size, seq_length, hidden_size)
        
        hidden_states = self.attention(hidden_states, mask)
        hidden_states = self.mlp(hidden_states)
        return hidden_states + residual


class LlamaModel(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str, num_layers: int, num_heads: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(hidden_size, intermediate_size, activation, num_heads) for _ in range(num_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, mask=None):
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.output_layer(hidden_states)
        return logits


import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import tqdm
import torch.nn.functional as F

# Load the vocabulary
with open("/kaggle/input/assign13-era-v3-dataset/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

id_to_token = {v: k for k, v in vocab.items()}

class Config:
    def __init__(self):
        self.vocab_size = 49152   # Vocabulary size
        self.hidden_size = 576     # Dimension of model embeddings
        self.intermediate_size = 1536  # Size of MLP layers
        self.num_layers = 30       # Number of decoder layers
        self.activation = "silu"   # Activation function
        self.num_heads = 8         # Number of attention heads (if needed)

class CyclicDataset(Dataset):
    def __init__(self, file_path, vocab, sample_size, num_tokens):
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = f.read()
        self.vocab = vocab
        self.sample_size = sample_size
        self.num_tokens = num_tokens
        self.tokens = self.tokenize(self.data)
        self.total_tokens = len(self.tokens)

    def tokenize(self, text):
        return [self.vocab.get(char, self.vocab["<|endoftext|>"]) for char in text]

    def __len__(self):
        return self.total_tokens // (self.sample_size * self.num_tokens)

    def __getitem__(self, idx):
        start = (idx * self.sample_size * self.num_tokens) % self.total_tokens
        batch_data = []
        for i in range(self.sample_size):
            sample_start = (start + i * self.num_tokens) % self.total_tokens
            sample_end = (sample_start + self.num_tokens) % self.total_tokens
            if sample_end < sample_start:
                sample = self.tokens[sample_start:] + self.tokens[:sample_end]
            else:
                sample = self.tokens[sample_start:sample_end]
            batch_data.append(sample)
        return torch.tensor(batch_data, dtype=torch.long)

# Training parameters
batch_size = 16
seq_length = 128
epochs = 100
learning_rate = 1e-4
save_every = 100
prediction_max_length = 50
input_text = "Hello, world!"

dataset = CyclicDataset("/kaggle/input/assign13-era-v3-dataset/input.txt", vocab, batch_size, seq_length)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Model Configuration
hidden_size = 576
intermediate_size = 1536
activation = "silu"
num_layers = 30

config = Config()
model = LlamaModel(
    hidden_size=config.hidden_size,
    intermediate_size=config.intermediate_size,
    activation=config.activation,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    vocab_size=config.vocab_size  # Pass vocab_size explicitly
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
criterion = nn.CrossEntropyLoss()

def save_checkpoint(epoch, batch_idx, model, optimizer, scheduler, path="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at batch {batch_idx} of epoch {epoch}.")

def generate_text(model, input_text, max_length=50, temperature=1.0, top_k=50, top_p=0.9):
    model.eval()
    input_ids = torch.tensor([vocab.get(char, vocab["<|endoftext|>"]) for char in input_text], dtype=torch.long).unsqueeze(0).to(device)
    generated_tokens = input_ids.tolist()[0]
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)[:, -1, :]

            # Check for NaN or Inf before proceeding
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("Error: Logits contain NaN or Inf values! Exiting.")
                print(logits)
                exit(1)

            # Apply temperature scaling
            logits = logits / temperature

            # Apply top-k sampling
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_value = values[:, -1].unsqueeze(-1)
                logits[logits < min_value] = -float("Inf")

            # Apply top-p (nucleus) sampling
            if top_p > 0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_logits[cumulative_probs > top_p] = -float("Inf")

                # Revert sorting order
                logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(dim=-1))

            # Sample from the modified logits distribution
            probabilities = F.softmax(logits, dim=-1)
            
            # Check for invalid probabilities
            if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
                print("Warning: Invalid probabilities encountered! Using argmax fallback.")
                next_token = torch.argmax(logits, dim=-1).squeeze().item()
            else:
                next_token = torch.multinomial(probabilities, num_samples=1).squeeze().item()

            # Ensure token index is within range
            if next_token < 0 or next_token >= len(id_to_token):
                print(f"Warning: Invalid token index {next_token}! Replacing with <|endoftext|>.")
                next_token = vocab["<|endoftext|>"]

            generated_tokens.append(next_token)
            input_ids = torch.tensor([generated_tokens], dtype=torch.long).to(device)

            # Stop generation if <|endoftext|> token is encountered
            if next_token == vocab["<|endoftext|>"]:
                break
    
    return ''.join(id_to_token.get(token, '?') for token in generated_tokens)



no_of_steps_to_break = 5000
steps = 0
GRADIENT_CLIP_VAL = 1.0

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        batch = batch.squeeze(0).to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        seq_length = inputs.shape[1]
        mask = torch.ones((inputs.shape[0], model.layers[0].attention.num_heads, seq_length, seq_length), dtype=torch.bool, device=inputs.device)

        outputs = model(inputs, mask=mask)


        logits = outputs.view(-1, len(vocab))

        loss = criterion(logits, targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
        
        if (batch_idx + 1) % save_every == 0:
            save_checkpoint(epoch, batch_idx, model, optimizer, scheduler)
            generated_text = generate_text(model, input_text, prediction_max_length)
            print(f"Generated text at batch {batch_idx}: {generated_text}")

        steps += 1
        if steps == no_of_steps_to_break:
            break
    
    scheduler.step()
    print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader)}")
    if steps == no_of_steps_to_break:
        break

print("Training complete!")
