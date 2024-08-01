import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from speech_tokenizer import SpeechTokenizer

# Constants
INPUT_DIR = '/home/hailey/Downloads/MusicBench'
DATA_DIR = '/media/hailey/More/AI/gpt2audio/music_data'
PREFIX = 'music_bench'
SHARD_SIZE = 5 * 1024 * 1024  # 5MB in bytes
CHUNK_LENGTH = 9  # seconds
SUB_CHUNK_LENGTH = 4.5  # seconds

# Initialize tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = SpeechTokenizer(device=device)


def process_audio(waveform, sample_rate):
    # Resample if necessary
    if sample_rate != tokenizer.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=tokenizer.sample_rate)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Create overlapping chunks
    audio_length = waveform.shape[1] / tokenizer.sample_rate
    tokenized_chunks = []

    for start_time in np.arange(0, audio_length - CHUNK_LENGTH + 1, 1):
        end_time = start_time + CHUNK_LENGTH
        chunk = waveform[:, int(start_time * tokenizer.sample_rate):int(end_time * tokenizer.sample_rate)]

        # Split into two SUB_CHUNK_LENGTH segments
        sub_chunks = torch.split(chunk, int(SUB_CHUNK_LENGTH * tokenizer.sample_rate), dim=1)

        for sub_chunk in sub_chunks:
            tokenized_sub_chunk = tokenizer.encode([sub_chunk])
            tokenized_chunks.append(tokenized_sub_chunk[0][:-1])  # Remove the last token as in the original script

    return tokenized_chunks


def get_next_shard_index(shard_type):
    existing_shards = [f for f in os.listdir(DATA_DIR) if f.startswith(f'{PREFIX}_{shard_type}_') and f.endswith('.npy')]
    if not existing_shards:
        return 0
    return max([int(f.split('_')[-1].split('.')[0]) for f in existing_shards]) + 1


def save_shard(shard, shard_index, shard_type):
    shard_path = os.path.join(DATA_DIR, f'{PREFIX}_{shard_type}_{shard_index:04d}.npy')
    np.save(shard_path, np.array(shard, dtype=np.int16))
    print(f"\nSaved {shard_type} shard: {shard_path}")
    return get_next_shard_index(shard_type)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    train_shard_index = get_next_shard_index('train')
    val_shard_index = get_next_shard_index('val')
    current_train_shard = []
    current_val_shard = []

    wav_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')]

    for wav_file in tqdm(wav_files, desc="Processing audio files"):
        file_path = os.path.join(INPUT_DIR, wav_file)
        waveform, sample_rate = torchaudio.load(file_path)

        # Skip if audio is shorter than CHUNK_LENGTH
        if waveform.shape[1] / sample_rate < CHUNK_LENGTH:
            continue

        tokenized_chunks = process_audio(waveform, sample_rate)

        for chunk in tokenized_chunks:
            if np.random.random() < 0.01:  # 1% chance for validation
                current_val_shard.extend(chunk)
                if len(current_val_shard) * 2 >= SHARD_SIZE:  # *2 because we're storing int16
                    val_shard_index = save_shard(current_val_shard, val_shard_index, 'val')
                    current_val_shard = []
            else:
                current_train_shard.extend(chunk)
                if len(current_train_shard) * 2 >= SHARD_SIZE:  # *2 because we're storing int16
                    train_shard_index = save_shard(current_train_shard, train_shard_index, 'train')
                    current_train_shard = []

    # Save any remaining data in the last shards
    if current_train_shard:
        save_shard(current_train_shard, train_shard_index, 'train')
    if current_val_shard:
        save_shard(current_val_shard, val_shard_index, 'val')


if __name__ == "__main__":
    main()
