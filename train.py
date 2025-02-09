import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import tqdm
from model import LlamaModel
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

        outputs = model(inputs)
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
