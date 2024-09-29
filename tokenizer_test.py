import torch
import torchaudio
from offset_tokenizer import AudioTokenizer
import numpy as np
from scipy.io import wavfile

audio_path = "/media/hailey/TVBox/music_dl/PMEDiA Music Pack 046 of 2024/Various Artists - Summer 2024 – Top 100 Songs (2024)/.unwanted/03. Benson boone - Beautiful Things.mp3"
seconds_to_load = 30
seconds_per_chunk = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the tokenizer
tokenizer = AudioTokenizer(device=device)

# Load a small clip from the audio file
waveform, sample_rate = torchaudio.load(audio_path, num_frames=seconds_to_load * tokenizer.sample_rate)

# Resample to 32kHz if necessary
if sample_rate != tokenizer.sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=tokenizer.sample_rate)
    waveform = resampler(waveform)

# Convert to mono if stereo
if waveform.size(0) > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Split the audio into chunks
chunk_size = int(seconds_per_chunk * tokenizer.sample_rate)
chunks = [waveform[:, i:i+chunk_size] for i in range(0, waveform.size(1), chunk_size)]

# Pad the last chunk if necessary
if chunks[-1].size(1) < chunk_size:
    chunks[-1] = torch.nn.functional.pad(chunks[-1], (0, chunk_size - chunks[-1].size(1)))

# Encode the audio chunks
encoded = tokenizer.encode(chunks)
for code in encoded:
    print(f"Code shape: {code.shape}")
    print(code[:18])
    print(code[-18:])

# Decode the audio
decoded = tokenizer.decode(encoded)

# Save the original and reconstructed audio
waveform_np = waveform.cpu().numpy()[0]  # Remove channel dimension
decoded_np = decoded.cpu().detach().numpy()

print(f"Original audio shape: {waveform_np.shape}")
print(f"Original audio dtype: {waveform_np.dtype}")
print(f"Original audio min: {waveform_np.min()}, max: {waveform_np.max()}")

print(f"Decoded audio shape: {decoded_np.shape}")
print(f"Decoded audio dtype: {decoded_np.dtype}")
print(f"Decoded audio min: {decoded_np.min()}, max: {decoded_np.max()}")

# Normalize to 16-bit range
waveform_np = np.int16(waveform_np / np.max(np.abs(waveform_np)) * 32767)

# For decoded audio, let's handle the shape issue and normalize
decoded_np = decoded_np.flatten()  # Flatten the 2D array to 1D
decoded_np = decoded_np - decoded_np.mean()  # Remove DC offset
max_val = np.max(np.abs(decoded_np))
if max_val > 0:
    decoded_np = decoded_np / max_val  # Normalize to [-1, 1]
decoded_np = np.int16(decoded_np * 32767)  # Convert to 16-bit int

print(f"Normalized original audio shape: {waveform_np.shape}")
print(f"Normalized original audio min: {waveform_np.min()}, max: {waveform_np.max()}")
print(f"Normalized decoded audio shape: {decoded_np.shape}")
print(f"Normalized decoded audio min: {decoded_np.min()}, max: {decoded_np.max()}")

wavfile.write("original_audio.wav", tokenizer.sample_rate, waveform_np)
wavfile.write("reconstructed_audio.wav", tokenizer.sample_rate, decoded_np)

print("Test completed. Check 'original_audio.wav' and 'reconstructed_audio.wav'.")