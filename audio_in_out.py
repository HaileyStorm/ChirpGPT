import torch
import torchaudio
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from torch.nn import functional as F
from scipy.io import wavfile
from gpt2 import GPT, GPTConfig
from speech_tokenizer import SpeechTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_and_prepare_audio(file_path, tokenizer):
    waveform, sample_rate = torchaudio.load(file_path)

    # Convert to mono if stereo
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 32kHz if necessary
    if sample_rate != tokenizer.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=tokenizer.sample_rate)
        waveform = resampler(waveform)

    # Trim to 4 seconds if longer
    max_length = 4 * tokenizer.sample_rate
    if waveform.size(1) > max_length:
        waveform = waveform[:, :max_length]

    target_length = 3 * tokenizer.sample_rate

    if waveform.size(1) > target_length:
        # Center crop
        start = (waveform.size(1) - target_length) // 2
        waveform = waveform[:, start:start + target_length]
    elif waveform.size(1) < target_length:
        # Pad
        padding_needed = target_length - waveform.size(1)
        max_end_padding = min(0.5 * tokenizer.sample_rate, padding_needed // 2)
        end_padding = int(max_end_padding)
        start_padding = padding_needed - end_padding
        waveform = F.pad(waveform, (start_padding, end_padding))

    return waveform


def tokenize_input(waveform, tokenizer):
    tokens = torch.from_numpy(tokenizer.encode([waveform])[0])
    return tokens


def generate_audio(model, input_tokens, tokenizer, num_return_sequences=1, max_new_tokens=512):
    input_tokens = input_tokens.unsqueeze(0).repeat(num_return_sequences, 1).to(device)

    with torch.no_grad():
        for _ in tqdm(range(max_new_tokens)):
            logits, _ = model(input_tokens[:, -model.config.block_size:])
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat([input_tokens, next_token], dim=1)

    return input_tokens[:, -(max_new_tokens+1):]  # Return only the newly generated tokens


def main():
    print(f"Using device: {device}")

    # Load model
    model = GPT(GPTConfig())
    state_dict = torch.load('./BIRD_FINAL_32khz_Small_NoTest_model_64432.pt', map_location=device)
    corrected_state_dict = OrderedDict(
        [(key.replace('_orig_mod.', ''), value) for key, value in state_dict['model'].items()])
    model.load_state_dict(corrected_state_dict)
    model.eval().to(device)

    # Initialize tokenizer
    tokenizer = SpeechTokenizer(device=device)

    # Load and prepare audio
    audio_path = "./tweet.mp3"  # Replace with your audio file path
    waveform = load_and_prepare_audio(audio_path, tokenizer)

    # Tokenize input
    input_tokens = tokenize_input(waveform, tokenizer)

    # Generate audio
    num_return_sequences = 4
    output_tokens = generate_audio(model, input_tokens, tokenizer, num_return_sequences)

    # Decode and save audio
    for i in range(num_return_sequences):
        audio_out = tokenizer.decode(np.array([output_tokens[i].tolist()]))
        wavfile.write(f'output_{i}.wav', tokenizer.sample_rate, audio_out.cpu().detach().numpy())


if __name__ == "__main__":
    main()