import torch
import torchaudio
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from torch.nn import functional as F
from scipy.io import wavfile
from gpt2 import GPT, GPTConfig
from speech_tokenizer import SpeechTokenizer
import sounddevice as sd
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


def record_audio(tokenizer, duration=3):
    print(f"Recording for {duration} seconds...")
    sample_rate = tokenizer.sample_rate
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("Recording finished.")

    # Convert to torch tensor
    waveform = torch.from_numpy(recording.T).float()

    # Ensure the recording is exactly 3 seconds
    target_length = 3 * sample_rate
    if waveform.size(1) > target_length:
        waveform = waveform[:, :target_length]
    elif waveform.size(1) < target_length:
        waveform = F.pad(waveform, (0, target_length - waveform.size(1)))

    #wavfile.write(f'recording.wav', tokenizer.sample_rate, recording)
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

    return input_tokens[:, -(max_new_tokens+1):]  # Return only the newly generated tokens (and the separator from the end of the input)


def main():
    print(f"Using device: {device}")

    # Load model
    model = GPT(GPTConfig())
    state_dict = torch.load('./log/model_s09000_vl4.9196.pt', map_location=device)
    corrected_state_dict = OrderedDict(
        [(key.replace('_orig_mod.', ''), value) for key, value in state_dict['model'].items()])
    model.load_state_dict(corrected_state_dict)
    model.eval().to(device)

    # Initialize tokenizer
    tokenizer = SpeechTokenizer(device=device)

    waveform = record_audio(tokenizer)

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