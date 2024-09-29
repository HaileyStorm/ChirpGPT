import torch
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from torch.nn import functional as F
from scipy.io import wavfile
from gpt2 import GPT, GPTConfig
from offset_tokenizer import AudioTokenizer
from dataloader import DataLoaderLite
import torch.distributed.checkpoint as dist_checkpoint
import random
import os

checkpoint_path = './log_small44khz/model_s05000_vl7.3514.pt' #'./log_edm/model_s50000_vl4.1830.pt'
shampoo = False

output_dir = './log_small44khz' #'./log_edm/50k/chunk3'
batch_size = 1 #3
prefill_chunks = 1
num_batches = 1
temperature = 0.96
top_k = 360


device = "cuda" if torch.cuda.is_available() else "cpu"
SUB_CHUNK_LENGTH = 10 #6
B = batch_size
T = 6483 #3072
chunk_size = 2161 #1024
input_length = prefill_chunks * SUB_CHUNK_LENGTH


def generate_audio(model, input_tokens, max_new_tokens=1024, temperature=0.96,
                   top_k=360):
    with torch.no_grad():
        for _ in tqdm(range(max_new_tokens)):
            # Get logits from the model
            logits, _ = model(input_tokens[:, -model.config.block_size:])
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Handle NaN and Inf values in logits
            nan_mask = torch.isnan(next_token_logits) | torch.isinf(next_token_logits)
            if nan_mask.any():
                #print("Warning: NaN or Inf values detected in logits. Replacing with very negative values.")
                next_token_logits = torch.where(nan_mask, torch.full_like(next_token_logits, -1e9), next_token_logits)

            # Compute softmax probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Perform top-k sampling
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)

            # Renormalize the top-k probabilities
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

            # Check for NaN values in top_k_probs
            if torch.isnan(top_k_probs).any():
                #print("Warning: NaN values detected in top-k probabilities. Using uniform distribution.")
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
            input_tokens = torch.cat([input_tokens, next_token], dim=1)

    return input_tokens[:, -(max_new_tokens+1):]  # Return only the newly generated tokens


def normalize_audio(audio):
    audio = audio.squeeze()  # Remove any extra dimensions
    audio = (audio - np.min(audio)) / (np.max(audio) - np.min(audio))  # Normalize to 0-1
    audio = (audio * 2) - 1  # Scale to -1 to 1
    audio = (audio * 32767).astype(np.int16)  # Scale to 16-bit integer range
    return audio


def main():
    print(f"Using device: {device}")

    # Load model
    model = GPT(GPTConfig(block_size=T, use_liger_gelu=True))
    if shampoo:
        chkpt = {}
        dist_checkpoint.load_state_dict(
            state_dict=chkpt,
            storage_reader=dist_checkpoint.FileSystemReader(checkpoint_path),
        )
        model_state_dict = OrderedDict([
            (key.replace('_orig_mod.', ''), value) for key, value in chkpt['model'].items()
        ])
        model.load_state_dict(model_state_dict)
    else:
        chkpt = torch.load(checkpoint_path, map_location=device)
        model_state_dict = OrderedDict([
            (key.replace('_orig_mod.', ''), value) for key, value in chkpt['model'].items()
        ])
        model.load_state_dict(model_state_dict)

    model.eval().to(device)

    # Initialize tokenizer
    tokenizer = AudioTokenizer(device=device)

    # Initialize dataloader
    val_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split="val", master_process=True, critical_divisor=chunk_size)

    # Skip random number of batches
    skip_batches = random.randint(0, val_loader.total_tokens // (B * T))
    val_loader.skip_batches(skip_batches)

    for b in range(num_batches):
        # Get batch from dataloader
        x_val, _ = val_loader.next_batch(False)

        # Generate prefill
        if input_length == SUB_CHUNK_LENGTH:  # chunk2+chunk3
            prefill = x_val[:batch_size][:, :chunk_size].to(device)
        else:  # chunk3
            prefill = x_val[:batch_size][:, :chunk_size*2].to(device)
        separators = torch.tensor([4097], dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        prefill = torch.cat([prefill, separators], dim=1)
        print(f"Prefill ({len(prefill[0])}): {prefill}")
        print(prefill[0][:18].tolist())
        print(prefill[0][-18:].tolist())

        # Generate samples
        output_tokens = generate_audio(model, prefill, max_new_tokens=int(T - (chunk_size * (input_length // SUB_CHUNK_LENGTH))),
                                       temperature=temperature, top_k=top_k)
        print(f"Output tokens ({output_tokens.shape}): {output_tokens}")

        # Save samples and prefill
        for i in range(batch_size):
            sample_name = os.path.join(output_dir, f'{input_length}sPrefill+{int(((T//chunk_size)*SUB_CHUNK_LENGTH) - input_length)}sSample_b{b}_s{i}.wav')
            prefill_audio = tokenizer.decode(np.array([prefill[i].tolist()]))
            sample_audio = tokenizer.decode(np.array([output_tokens[i].tolist()]))
            save_audio = np.append(prefill_audio.cpu().detach().numpy(), sample_audio.cpu().detach().numpy())
            normalized_audio = normalize_audio(save_audio)
            wavfile.write(sample_name, tokenizer.sample_rate, normalized_audio)


if __name__ == "__main__":
    main()