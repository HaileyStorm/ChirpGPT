import torch
import torchaudio
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from torch.nn import functional as F
from scipy.io import wavfile
from gpt2 import GPT, GPTConfig
from two_sep_tokenizer import AudioTokenizer
import torch.distributed.checkpoint as dist_checkpoint

checkpoint_path = './log/model_s110500_vl4.90655.pt'
shampoo = False

audio_paths = [
    "/media/hailey/TVBox/music_dl/PMEDiA Music Pack 046 of 2024/Various Artists - Summer 2024 – Top 100 Songs (2024)/.unwanted/03. Benson boone - Beautiful Things.mp3",
    "/media/hailey/TVBox/music_dl/PMEDiA Music Pack 046 of 2024/Various Artists - Summer 2024 – Top 100 Songs (2024)/.unwanted/02. David guetta & Onerepublic - I Don't Wanna Wait.mp3",
    "/media/hailey/TVBox/music_dl/PMEDiA Music Pack 046 of 2024/Various Artists - Summer 2024 – Top 100 Songs (2024)/.unwanted/01. Dua lipa - Training Season.mp3",
    ]
start_time = 46  # Seconds
input_length = 6  # Seconds
assert input_length == 6 or input_length == 12
batch_size = 3
num_batches = 3
temperature = 0.96
top_k = 360

device = "cuda" if torch.cuda.is_available() else "cpu"
SUB_CHUNK_LENGTH = 6


def load_and_prepare_audio(file_path, start_time, input_length, tokenizer):
    waveform, sample_rate = torchaudio.load(file_path)

    # Convert to mono if stereo
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 32kHz if necessary
    if sample_rate != tokenizer.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=tokenizer.sample_rate)
        waveform = resampler(waveform)

    # Ensure at least SUB_CHUNK_LENGTH*3 seconds of audio remain after the start time
    if waveform.size(1) - int(start_time * tokenizer.sample_rate) < int(SUB_CHUNK_LENGTH * 3 * tokenizer.sample_rate):
        raise ValueError(f"Audio is too short after start time. Start time: {start_time}, Required remaining length: {SUB_CHUNK_LENGTH * 3}")

    # Slice audio from the start time
    start = int(start_time * tokenizer.sample_rate)
    end = start + int(input_length * tokenizer.sample_rate)
    waveform = waveform[:, start:end]

    return waveform


def tokenize_input(waveform, tokenizer):
    tokens = torch.from_numpy(tokenizer.encode([waveform])[0])
    return tokens


def generate_audio(model, input_tokens, num_return_sequences=1, max_new_tokens=1024, temperature=0.96,
                   top_k=360):
    input_tokens = input_tokens.unsqueeze(0).repeat(num_return_sequences, 1).to(device)

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


def main():
    print(f"Using device: {device}")

    # Load model
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
        chkpt = torch.load(checkpoint_path, map_location=device)
        # Load model state
        model_state_dict = OrderedDict([
            (key.replace('_orig_mod.', ''), value) for key, value in chkpt['model'].items()
        ])
        model.load_state_dict(model_state_dict)

    model.eval().to(device)

    # Initialize tokenizer
    tokenizer = AudioTokenizer(device=device)

    for b in range(num_batches):
        # Load and prepare audio
        waveform = load_and_prepare_audio(audio_paths[b % len(audio_paths)], start_time, input_length, tokenizer)

        # Tokenize input
        input_tokens = tokenize_input(waveform, tokenizer)

        # Generate audio
        output_tokens = generate_audio(model, input_tokens, batch_size, max_new_tokens=2048 if input_length == 6 else 1024, temperature=temperature, top_k=top_k)

        # Decode and save audio
        for i in range(batch_size):
            print(np.array([output_tokens[i].tolist()]))
            print(len(output_tokens[i].tolist()))
            audio_out = tokenizer.decode(np.array([output_tokens[i].tolist()]))
            wavfile.write(f'./log/final/{"chunk2and3" if input_length == 6 else "chunk3"}_{i}.wav', tokenizer.sample_rate, audio_out.cpu().detach().numpy())


if __name__ == "__main__":
    main()
