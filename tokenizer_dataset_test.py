import torch
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from offset_tokenizer import AudioTokenizer
from dataloader import DataLoaderLite
import os

output_dir = './decode_test'
batch_size = 1  # Keep this at 1 for now
num_batches = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

T = 6483
chunk_size = 2161


def normalize_audio(audio):
    audio = audio.squeeze()  # Remove any extra dimensions
    audio = (audio - np.min(audio)) / (np.max(audio) - np.min(audio))  # Normalize to 0-1
    audio = (audio * 2) - 1  # Scale to -1 to 1
    audio = (audio * 32767).astype(np.int16)  # Scale to 16-bit integer range
    return audio


def main():
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = AudioTokenizer(device=device)

    # Initialize dataloader
    val_loader = DataLoaderLite(B=batch_size, T=T, process_rank=0, num_processes=1, split="val", master_process=True,
                                critical_divisor=chunk_size)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for b in tqdm(range(num_batches), desc="Processing batches"):
        # Get batch from dataloader
        x_val, _ = val_loader.next_batch(False)

        for chunks in [1, 2, 3]:
            # Prepare input for decoding
            input_tokens = x_val[:, :chunk_size * chunks].to(device)

            # Add separator token at the end
            separators = torch.tensor([4097], dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
            input_tokens = torch.cat([input_tokens, separators], dim=1)

            print(f"\nBatch {b}, Chunks {chunks}")
            print(f"Input tokens shape: {input_tokens.shape}")
            print(f"First 18 tokens: {input_tokens[0][:18].tolist()}")
            print(f"Last 18 tokens: {input_tokens[0][-18:].tolist()}")

            try:
                # Decode audio
                decoded_audio = tokenizer.decode(input_tokens.cpu().numpy())
                print(decoded_audio)

                # Save decoded audio
                for i in range(batch_size):
                    sample_name = os.path.join(output_dir, f'batch{b}_chunks{chunks}_sample{i}.wav')
                    normalized_audio = normalize_audio(decoded_audio[i].cpu().numpy())
                    wavfile.write(sample_name, tokenizer.sample_rate, normalized_audio)

                print(f"Successfully decoded and saved audio for {chunks} chunks")
            except Exception as e:
                print(f"Error decoding audio for {chunks} chunks: {str(e)}")


if __name__ == "__main__":
    main()