from gpt2 import GPT, GPTConfig
import torch, torchaudio
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from two_sep_tokenizer import AudioTokenizer
from scipy.io.wavfile import write
import torch.distributed.checkpoint as dist_checkpoint
import math

checkpoint_path = './log/model_s110500_vl4.90655.pt'
shampoo = False

batch_size = 3
num_batches = 15
# Generally OK ~0.9-1.01, depending on what you're after, best ~0.935-0.975, default 0.96
temperature = 0.96
# Best 640-720, default 712
top_k_max = 712
# OK ~256-512, best ~350-385, default 360
top_k_min = 360
# Number of generated tokens to take to decrease from top_k_max to top_k_min
# Depending on min/max, OK 512 on down to 128 or less, best ~384-416, default 408
top_k_warmup = 408

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
model = GPT(GPTConfig(block_size=3072))

if shampoo:
    chkpt = {}
    dist_checkpoint.load_state_dict(
        state_dict=chkpt,
        storage_reader=dist_checkpoint.FileSystemReader(checkpoint_path),
    )

    # Load model state
    model_state_dict = OrderedDict([
        (key.replace('_orig_mod.', ''), value) for key, value in chkpt['model'].items()
    ])
    model.load_state_dict(model_state_dict)
else:
    chkpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # Load model state
    model_state_dict = OrderedDict([
        (key.replace('_orig_mod.', ''), value) for key, value in chkpt['model'].items()
    ])
    model.load_state_dict(model_state_dict)

model.eval()
model.to(device)

seperator = 4097

# 3s @ 32khz = 512 tokens
# 4.5s = 768 tokens
# 6s = 1024
# We could reduce this by one and append the final separator token manually.
max_length = 3072

sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(1337)


def get_top_k(step):
    if step < top_k_warmup:
        progress = step / top_k_warmup
        # Use a sigmoid function for a gradual start and more rapid finish
        sigmoid_progress = 1 / (1 + math.exp(-10 * (progress - 0.5)))
        return int(top_k_max - sigmoid_progress * (top_k_max - top_k_min))
    else:
        return top_k_min


for b in range(num_batches):
    tokens = [seperator]
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(batch_size, 1)
    xgen = tokens.to(device)
    output_tokens = []

    # Initialize the progress bar
    with tqdm(total=max_length, desc="Generating...") as pbar:
        # Set initial progress
        pbar.update(xgen.size(1))

        while xgen.size(1) <= max_length:
            # Calculate current top_k value
            current_step = min(xgen.size(1) - 1, top_k_warmup)
            top_k = get_top_k(current_step)
            pbar.set_description(f"Generating (top_k={top_k})...", refresh=True)

            # forward the model to get the logits
            with torch.no_grad():
                # Get logits from the model
                logits, _ = model(xgen[:, -model.config.block_size:])
                next_token_logits = logits[:, -1, :]

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Handle NaN and Inf values in logits
                nan_mask = torch.isnan(next_token_logits) | torch.isinf(next_token_logits)
                if nan_mask.any():
                    # print("Warning: NaN or Inf values detected in logits. Replacing with very negative values.")
                    next_token_logits = torch.where(nan_mask, torch.full_like(next_token_logits, -1e9), next_token_logits)

                # Compute softmax probabilities
                probs = F.softmax(next_token_logits, dim=-1)

                # Perform top-k sampling
                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)

                # Renormalize the top-k probabilities
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

                # Check for NaN values in top_k_probs
                if torch.isnan(top_k_probs).any():
                    # print("Warning: NaN values detected in top-k probabilities. Using uniform distribution.")
                    top_k_probs = torch.ones_like(top_k_probs) / top_k

                # Sample from the top-k distribution
                try:
                    sample_indices = torch.multinomial(top_k_probs, num_samples=1)
                    next_token = torch.gather(top_k_indices, -1, sample_indices)
                except RuntimeError as e:
                    print(f"Error during sampling: {e}")
                    print("Falling back to argmax selection from top-k.")
                    next_token = top_k_indices[:, 0].unsqueeze(-1)  # Select the highest probability token

                # Append the new token to the sequence
                xgen = torch.cat([xgen, next_token], dim=1)

                pbar.update(1)  # Update by the number of new tokens added
        # print the generated text
        for i in range(batch_size):
            tokens = xgen[i, :max_length+1].tolist()
            output_tokens.append(tokens)

    with torch.no_grad():
        del xgen
        torch.cuda.empty_cache()
    tokenizer = AudioTokenizer(device=device)


    for i in range(batch_size):
        print(np.array([output_tokens[i]]))
        audio_out = tokenizer.decode(np.array([output_tokens[i]]))
        write(f'./log/final/tk712_360_408_round3/full_b{b}_{i}.wav', tokenizer.sample_rate, audio_out.cpu().detach().numpy())

    with torch.no_grad():
        del output_tokens
        del tokenizer
        torch.cuda.empty_cache()
