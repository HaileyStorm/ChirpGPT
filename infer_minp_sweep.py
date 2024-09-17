from gpt2 import GPT, GPTConfig
import torch
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from two_sep_tokenizer import AudioTokenizer
from scipy.io.wavfile import write
import os

checkpoint_path = './log_edm/model_s50000_vl4.1830.pt'
batch_size = 2
num_batches = 5
max_length = 3072
seed = 53162  # 42 (but really, at least for the EDM model lol)

# Recommend p_base 0.045-0.055
# Recommend temp 0.965-0.99 (for p_base 0.045), but can get fun results up to 1.1 or more

# Lists of p_base and min_p_temp values to explore
p_base_values = [0.045, 0.05, 0.055, 0.06]
min_p_temp_values = [0.96, 0.97, 0.98, 0.99, 1.0, 1.02, 1.04, 1.06, 1.09, 1.12]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

seperator = 4097

model = GPT(GPTConfig(block_size=3072))
chkpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model_state_dict = OrderedDict([
    (key.replace('_orig_mod.', ''), value) for key, value in chkpt['model'].items()
])
model.load_state_dict(model_state_dict)
model.eval()
model.to(device)


def min_p_sampling(logits, p_base):
    probs = F.softmax(logits, dim=-1)
    p_top = probs.max()
    p_threshold = p_base * p_top
    mask = probs >= p_threshold
    filtered_probs = probs * mask
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    return filtered_probs


def generate_samples(p_base, min_p_temp, batch_seed):
    torch.manual_seed(batch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(batch_seed)

    initial_tokens = torch.tensor([seperator], dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    xgen = initial_tokens.to(device)
    output_tokens = []

    with tqdm(total=max_length, desc=f"Generating (p_base={p_base}, temp={min_p_temp})...") as pbar:
        pbar.update(xgen.size(1))

        while xgen.size(1) <= max_length:
            with torch.no_grad():
                logits, _ = model(xgen[:, -model.config.block_size:])
                next_token_logits = logits[:, -1, :] / min_p_temp

                nan_mask = torch.isnan(next_token_logits) | torch.isinf(next_token_logits)
                if nan_mask.any():
                    next_token_logits = torch.where(nan_mask, torch.full_like(next_token_logits, -1e9),
                                                    next_token_logits)

                filtered_probs = min_p_sampling(next_token_logits, p_base)

                if torch.isnan(filtered_probs).any():
                    filtered_probs = torch.ones_like(filtered_probs) / filtered_probs.shape[-1]

                try:
                    next_token = torch.multinomial(filtered_probs, num_samples=1)
                except RuntimeError as e:
                    print(f"Error during sampling: {e}")
                    print("Falling back to argmax selection.")
                    next_token = filtered_probs.argmax(dim=-1).unsqueeze(-1)

                xgen = torch.cat([xgen, next_token], dim=1)
                pbar.update(1)

        for i in range(batch_size):
            tokens = xgen[i, :max_length + 1].tolist()
            output_tokens.append(tokens)

    return output_tokens


def save_audio(tokens, p_base, min_p_temp, batch, sample_index):
    tokenizer = AudioTokenizer(device=device)
    audio_out = tokenizer.decode(np.array([tokens]))

    output_dir = f'./log_edm/50k/minp_test/p_base_{p_base}'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f'temp_{min_p_temp}_b{batch}_{sample_index}.wav')
    write(output_path, tokenizer.sample_rate, audio_out.cpu().detach().numpy())


# Main loop
for batch in range(num_batches):
    batch_seed = seed + batch

    for p_base in p_base_values:
        for min_p_temp in min_p_temp_values:
            #if (p_base >= 0.12 and min_p_temp < 0.97) or (p_base >= 0.1 and min_p_temp < 0.9) or (p_base >= 0.1 and min_p_temp > 1.5) or (p_base >= 0.06667 and min_p_temp > 1.4) or (p_base < 0.06667 and min_p_temp > 1.2):
             #   continue

            output_tokens = generate_samples(p_base, min_p_temp, batch_seed)

            for i, tokens in enumerate(output_tokens):
                save_audio(tokens, p_base, min_p_temp, batch, i)

    # Clear CUDA cache after each batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("Generation complete!")