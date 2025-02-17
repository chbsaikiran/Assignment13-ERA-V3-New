import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import math
import os
from numpy import arange
import torch.nn.functional as F
from torch.optim import AdamW

# Import the LlamaModel from model_manual.py
from model import LlamaForCausalLM

#def generate_text(model, input_text, vocab, id_to_token, device, max_length=50, temperature=0.7):
#    model.eval()
#    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
#    if tokenizer.pad_token is None:
#        if tokenizer.eos_token:
#            tokenizer.pad_token = tokenizer.eos_token
#        else:
#            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
#            tokenizer.resize_token_embeddings(len(tokenizer))
#
#    input_ids = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).input_ids.to(device)
#    attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(device)
#    generated_tokens = input_ids.tolist()[0]
#
#    with torch.no_grad():
#        for _ in range(max_length):
#            logits = model(input_ids)[:, -1, :]
#            logits = logits / temperature
#            probabilities = F.softmax(logits, dim=-1)
#            probabilities = probabilities[-1]
#            next_token = torch.multinomial(probabilities, num_samples=1).item()
#
#            if next_token == tokenizer.eos_token_id:
#                break
#
#            generated_tokens.append(next_token)
#            input_ids = torch.tensor([generated_tokens], dtype=torch.long).to(device)
#            attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(device)
#
#    model.train()
#    return tokenizer.decode(generated_tokens, skip_special_tokens=True)
def generate_text(
    model, prompt, max_length=50, temperature=0.7, top_k=50
):
    model.eval()
    device = model.device
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.resize_token_embeddings(len(tokenizer))

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature

            # Apply top-k sampling
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = torch.softmax(top_k_logits, dim=-1)

            # Sample from the filtered distribution
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[0, next_token_idx[0]]

            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    model.train()
    return generated_text


class DataloaderLite:
    def __init__(self, file_path, seq_len, batch_size):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer", add_prefix_space=True)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.tokenizer.resize_token_embeddings(len(self.tokenizer))

        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.epochs = len(self.text) // (self.seq_len * self.batch_size)
        self.current_position = 0
        self.padded_chunks = []
        self.max_len = 0
        
    def get_max_length(self):
        return self.max_len

    def next_batch(self):

        self.chunks = [self.text[(self.current_position + i):(self.current_position + i + self.seq_len)] for i in range(0, self.seq_len*self.batch_size, self.seq_len)]
        self.current_position = self.current_position + self.seq_len*self.batch_size
        if self.current_position + (self.seq_len*self.batch_size + 1) > len(self.text):
            self.current_position = 0
        self.encoded_chunks = [self.tokenizer.encode(chunk, return_tensors='pt', truncation=True, max_length=self.seq_len) for chunk in self.chunks]

        self.max_len = (self.next_power_of_2(max(chunk.shape[1] for chunk in self.encoded_chunks))) + 2
        self.padded_chunks = []
        attention_mask = torch.ones((self.batch_size, self.max_len), dtype=torch.long)
        count = 0
        for chunk in self.encoded_chunks:
            input_ids = torch.cat((chunk, torch.full((1, self.max_len - chunk.shape[1]), self.tokenizer.pad_token_id, dtype=torch.long)), dim=1)
            attention_mask[count, :] = torch.cat((attention_mask[count, :chunk.shape[1]], torch.zeros(self.max_len - chunk.shape[1], dtype=torch.long)))
            self.padded_chunks.append(input_ids)
            self.padded_chunks.append(attention_mask[count,:])
            count = count + 1
        return self.padded_chunks

    def next_power_of_2(self,n):
        if n <= 0:
            return 1  # For non-positive numbers, return 1 as the next power of 2
        
        power = 1
        while power <= n:
            power <<= 1  # Left shift is equivalent to multiplying by 2
        return power

# Random initialization
def init_weights(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)  # Normal initialization with low variance
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.LayerNorm) or hasattr(m, 'eps'):
        torch.nn.init.normal_(m.weight, mean=1.0, std=0.02)

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,}")

def train_model(config, train_file, steps, output_dir):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.resize_token_embeddings(len(tokenizer))

    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}

    dataloader = DataloaderLite(train_file, SEQ_LEN, BATCH_SIZE)
    #padded_chunks = dataloader.next_batch()
    #print(padded_chunks[0])
    #print(padded_chunks[1])
    #print(padded_chunks[2])
    #print(padded_chunks[3])

    model = LlamaForCausalLM(config)
    model.apply(init_weights)
    #for p in model.parameters():
    #    p.data.clamp_(-1e5, 1e5)
    #print_model_parameters(model)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.device = next(model.parameters()).device
    
    LEARNING_RATE = 1e-4
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    total_steps=10000,
    pct_start=0.1,
    anneal_strategy="cos",
    cycle_momentum=False,
    )
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    progress_bar = tqdm(range(steps), desc="Training", dynamic_ncols=True, position=0, leave=True)
    
    PROMPT = "inventory to particularise their abundance"
    
    step = 0
    while step < steps:
        input_tokens = dataloader.next_batch()
        max_len = dataloader.get_max_length()
        input_list = []
        attenttion_list = []
        for i in arange(BATCH_SIZE):
            input_list.append(input_tokens[2*i])
            attenttion_list.append(input_tokens[2*i+1])
    
        #print(input_list[0].shape)
        input_ids = torch.vstack(input_list)
        #print(input_ids.shape)
        inputs = input_ids[:, :-2]  # Keep batch dim and remove last token
        targets = input_ids[:, 1:-1]  # Keep batch dim and remove first token
        #print(inputs[:3,:])
        #print(inputs.shape)
        attention_mask = torch.vstack(attenttion_list)
        attentions = attention_mask[:, :-2]
        device = next(model.parameters()).device
        inputs, attentions = inputs.to(device), attentions.to(device)
        #positions = torch.arange(0, inputs.size(1), dtype=torch.long).unsqueeze(0).repeat(inputs.size(0), 1).to(device)
        #print(inputs.shape)
        #print(attentions.shape)
        #print(positions.shape)
        
        optimizer.zero_grad()
        logits = model(inputs)
        
        labels = targets.to(device)  # Move labels to the same device as the model and inputs
        # Create a mask based on counts
        mask = attentions.bool()  # The attention mask already has the correct shape
        #logits = logits.transpose(0, 1)  # Align logits to (batch_size, seq_len, vocab_size)
        logits_masked = logits[mask].contiguous().view(-1, config.vocab_size)
        labels_masked = labels[mask].contiguous().view(-1)
        probabilities = F.softmax(logits_masked, dim=-1)
        max_prob_indices = torch.argmax(probabilities, dim=-1)
        #print(max_prob_indices[:10])
        #print(labels_masked[:10])
        
        if mask.sum() == 0:
            raise ValueError("Attention mask sums to zero!")
        
        loss_cross_entropy = F.cross_entropy(logits_masked, labels_masked, label_smoothing=0.1)
        #loss = loss_fn(logits_masked, labels_masked)
        
        loss_cross_entropy.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        
        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss_cross_entropy.item(), refresh=True)
        
        if step % 250 == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict()
            }, os.path.join(output_dir, f'checkpoint.pt'))
            progress_bar.clear()  # Clear progress bar before printing text
            #generated_text = generate_text(model, PROMPT, vocab, id_to_token, model.device)
            generated_text = generate_text(
                            model,
                            PROMPT,
                            temperature=0.7,
                            max_length=100,  # Increased max length
                        )
            print(f"Generated text at step {step}: {generated_text}")  # Use print to avoid tqdm interference
            progress_bar.refresh()  # Refresh the progress bar after printing
        
        
        step += 1
        if step >= steps:
            break
    
    torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict()
            }, os.path.join(output_dir, f'checkpoint.pt'))
    
    progress_bar.close()  # Ensure the progress bar closes cleanly

class Config:
    pass

if __name__ == "__main__":
    config = Config()
    config.vocab_size = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer").vocab_size
    config.num_layers = 30
    config.hidden_size = 576
    config.num_attention_heads = 8
    config.rms_norm_eps = 1.0e-05
    config.max_position_embeddings = 2048
    config.rope_theta = 500000.0
    config.hidden_act = False
    config.intermediate_size = 1536
    config.rope_interleaved = False
    #config.rope_scaling = null
    config.rope_theta = 10000.0

    BATCH_SIZE = 8
    SEQ_LEN = 256

    train_model(config, '/kaggle/input/assign13-era-v3-dataset/input.txt', 5000, './output')
    #train_model(config, '/content/input.txt', 5000, './output')
