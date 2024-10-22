import torch
import torchaudio
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from torch.nn import functional as F
from scipy.io import wavfile
from gpt2 import GPT, GPTConfig
from two_sep_tokenizer import AudioTokenizer
import sounddevice as sd
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

SUB_CHUNK_LENGTH = 6  # 4.5
TOKENS_TO_PREDICT = 2048  # 768


def record_audio(tokenizer, duration=SUB_CHUNK_LENGTH):
    print(f"Recording for {duration} seconds...")
    sample_rate = tokenizer.sample_rate
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("Recording finished.")

    # Convert to torch tensor
    waveform = torch.from_numpy(recording.T).float()

    # Ensure the recording is exactly duration seconds
    target_length = duration * sample_rate
    if waveform.size(1) > target_length:
        waveform = waveform[:, :target_length]
    elif waveform.size(1) < target_length:
        waveform = F.pad(waveform, (0, target_length - waveform.size(1)))

    #wavfile.write(f'recording.wav', tokenizer.sample_rate, recording)
    return waveform


def tokenize_input(waveform, tokenizer):
    tokens = torch.from_numpy(tokenizer.encode([waveform])[0])
    return tokens


def generate_audio(model, input_tokens, num_return_sequences=1, max_new_tokens=TOKENS_TO_PREDICT, temperature=0.95,
                   top_k=650):
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
    model = GPT(GPTConfig())
    state_dict = torch.load('./log/model_s12900_vl4.7095.pt', map_location=device)
    corrected_state_dict = OrderedDict(
        [(key.replace('_orig_mod.', ''), value) for key, value in state_dict['model'].items()])
    model.load_state_dict(corrected_state_dict)
    model.eval().to(device)

    # Initialize tokenizer
    tokenizer = AudioTokenizer(device=device)

    waveform = record_audio(tokenizer)

    # Tokenize input
    input_tokens = tokenize_input(waveform, tokenizer)

    # Generate audio
    num_return_sequences = 3
    output_tokens = generate_audio(model, input_tokens, num_return_sequences)

    # Decode and save audio
    for i in range(num_return_sequences):
        audio_out = tokenizer.decode(np.array([output_tokens[i].tolist()]))
        wavfile.write(f'output_{i}.wav', tokenizer.sample_rate, audio_out.cpu().detach().numpy())


if __name__ == "__main__":
    main()
