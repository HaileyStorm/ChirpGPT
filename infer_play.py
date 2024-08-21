from gpt2 import GPT, GPTConfig
import torch, torchaudio
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from speech_tokenizer import SpeechTokenizer
from scipy.io.wavfile import write

checkpoint_path = './log/model_s71600_vl5.1429.pt'
shampoo = False

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
model = GPT(GPTConfig(block_size=3072))

if shampoo:
    state_dict = {}
    torch.distributed.checkpoint.load_state_dict(
        state_dict=state_dict,
        storage_reader=torch.distributed.checkpoint.FileSystemReader(checkpoint_path),
    )

    # Load model state
    model_state_dict = OrderedDict([
        (key.replace('_orig_mod.', ''), value) for key, value in state_dict['model'].items()
    ])
    model.load_state_dict(model_state_dict)
else:
    original_state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # Corrected state dictionary
    state_dict = {
        'model': OrderedDict([
            (key.replace('_orig_mod.', ''), value) for key, value in original_state_dict['model'].items()
        ]),
        'config': original_state_dict['config'],
        'step': original_state_dict['step'],
        'val_loss': original_state_dict['val_loss']
    }
    model.load_state_dict(state_dict['model'])

model.eval()
model.to(device)

seperator = 4097

num_return_sequences = 3
# 3s @ 32khz = 512 tokens
# 4.5s = 768 tokens
# 6s = 1024
# We could reduce this by one and append the final separator token manually.
max_length = 3072  # 1536

tokens = [seperator]
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(1337)
temperature = 0.95
top_k = 650

output_tokens = []

# Initialize the progress bar
with tqdm(total=max_length) as pbar:
    # Set initial progress
    pbar.update(xgen.size(1))
    
    while xgen.size(1) <= max_length:
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
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length+1].tolist()
        output_tokens.append(tokens)

with torch.no_grad():
    del xgen
    torch.cuda.empty_cache()
tokenizer = SpeechTokenizer(device=device)


def find_last_instance_of_seperator(lst, element=seperator):
    reversed_list = lst[::-1]
    try:
        reversed_index = reversed_list.index(element)
        return len(lst) - 1 - reversed_index
    except ValueError:
        raise ValueError


for i in range(num_return_sequences):
    print(np.array([output_tokens[i]]))
    #print(np.array([output_tokens[i][:find_last_instance_of_seperator(output_tokens[i]) + 1]]))
    #audio_out = tokenizer.decode(np.array([output_tokens[i][:find_last_instance_of_seperator(output_tokens[i]) + 1]]))
    audio_out = tokenizer.decode(np.array([output_tokens[i]]))
    write(f'test_{i}.wav', tokenizer.sample_rate, audio_out.cpu().detach().numpy())
