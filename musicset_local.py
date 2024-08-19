import os
from random import shuffle
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import json
from two_sep_tokenizer import AudioTokenizer

# Constants
INPUT_DIR = '/media/hailey/TVBox/music_dl'
DATA_DIR = '/media/hailey/More/AI/gpt2audio/music_data'
PREFIX = 'music'
SHARD_SIZE = 15 * 1024 * 1024  # 15MB in bytes
CHUNK_LENGTH = 18  # seconds
SUB_CHUNK_LENGTH = 6  # seconds
# The bigger this is, the more empty data will be tokenized (each file is padded to be divisible by this)
# With current main dataset, 10 = 36 hours to tokenize (1 = something like 12 days iirc)
SECONDS_PER_STEP = 7  # seconds
BATCH_SIZE = 3

assert SECONDS_PER_STEP <= CHUNK_LENGTH

# Initialize tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AudioTokenizer(device=device)

state_file = os.path.join(DATA_DIR, 'shuffle_state.json')


def save_shuffle_state(audio_files, processed_count):
    state = {
        'shuffled_files': audio_files,
        'processed_count': processed_count
    }
    with open(state_file, 'w') as f:
        json.dump(state, f)


def load_shuffle_state():
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
        return state['shuffled_files'], state['processed_count']
    return None, None


def pad_audio(waveform, sample_rate):
    audio_length = waveform.shape[1]
    samples_per_chunk = int(CHUNK_LENGTH * sample_rate)

    # Ensure the audio is at least samples_per_chunk long
    if audio_length < samples_per_chunk:
        pad_size = samples_per_chunk - audio_length
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))

    return waveform


def process_audio(waveforms, sample_rates):
    # Resample and convert to mono if necessary
    processed_waveforms = []
    for waveform, sample_rate in zip(waveforms, sample_rates):  #tqdm(zip(waveforms, sample_rates), "\tResampling and padding batch...", dynamic_ncols=True, total=len(waveforms)):
        if sample_rate != tokenizer.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=tokenizer.sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        processed_waveforms.append(pad_audio(waveform, tokenizer.sample_rate))

    # Calculate chunk sizes in samples
    samples_per_chunk = int(CHUNK_LENGTH * tokenizer.sample_rate)
    samples_per_subchunk = int(SUB_CHUNK_LENGTH * tokenizer.sample_rate)

    tokenized_chunks = [[] for _ in range(len(waveforms))]
    max_length = max(w.shape[1] for w in processed_waveforms)
    # Allow one subchunk to be mostly, but not entirely, silence (allow up to almost one subchunk of silence per song)
    max_pad = samples_per_chunk // 3.1

    for start_time in range(0, max_length - samples_per_chunk + 1, tokenizer.sample_rate * SECONDS_PER_STEP):  #tqdm(range(0, max_length - samples_per_chunk + 1, tokenizer.sample_rate * SECONDS_PER_STEP), "\tChunking and tokenizing batch...", dynamic_ncols=True):
        end_time = start_time + samples_per_chunk
        batch_chunks = []
        for w in processed_waveforms:
            if w.shape[1] >= end_time - max_pad:
                if w.shape[1] < end_time:
                    pad_size = end_time - w.shape[1]
                    padded_w = torch.nn.functional.pad(w, (0, pad_size))
                else:
                    padded_w = w
                batch_chunks.append(padded_w[:, start_time:end_time])

        if not batch_chunks:
            break

        sub_chunks = [torch.split(chunk, samples_per_subchunk, dim=1) for chunk in batch_chunks]
        valid_sub_chunks = [sc for sc in sub_chunks if
                            len(sc) == 3 and all(s.shape[1] == samples_per_subchunk for s in sc)]

        if not valid_sub_chunks:
            continue

        tokenized_sub_chunks = []
        for i in range(3):  # For each subchunk position
            batch_subchunks = [sc[i] for sc in valid_sub_chunks]
            tokenized_batch = tokenizer.encode(batch_subchunks)
            tokenized_sub_chunks.append([t[:-1] for t in tokenized_batch])

        for i, (sc1, sc2, sc3) in enumerate(zip(*tokenized_sub_chunks)):
            if all(len(sc) == 1024 for sc in (sc1, sc2, sc3)):
                tokenized_chunks[i].append([sc1, sc2, sc3])

    return [chunk for chunk in tokenized_chunks if chunk]  # Remove empty lists


def get_next_shard_index(shard_type):
    existing_shards = [f for f in os.listdir(DATA_DIR) if
                       f.startswith(f'{PREFIX}_{shard_type}_') and f.endswith('.npy')]
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

    shuffled_files, processed_count = load_shuffle_state()
    if shuffled_files is None:
        print("Finding files...")
        audio_files = []
        for root, dirs, files in os.walk(INPUT_DIR):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    audio_files.append(os.path.join(root, file))
        print(f"Found {len(audio_files)} files.\n")
        shuffle(audio_files)
        processed_count = 0
    else:
        audio_files = shuffled_files
        print(f"Resuming from file {processed_count} of {len(audio_files)}\n")

    total_chunks = 0
    bar = tqdm(range(processed_count, len(audio_files), BATCH_SIZE), initial=processed_count//BATCH_SIZE, total=len(audio_files) // BATCH_SIZE, desc="Processing audio files", dynamic_ncols=True, unit="batch")
    try:
        for i in bar:
            batch_files = audio_files[i:i + BATCH_SIZE]
            waveforms = []
            sample_rates = []
            for file_path in batch_files:  # tqdm(batch_files, desc="\tLoading audio data...", dynamic_ncols=True):
                try:
                    waveform, sample_rate = torchaudio.load(file_path)
                    if waveform.shape[1] / sample_rate >= CHUNK_LENGTH:
                        waveforms.append(waveform)
                        sample_rates.append(sample_rate)
                except:
                    print(f"Error opening {file_path}")

            if not waveforms:
                continue

            tokenized_chunks = process_audio(waveforms, sample_rates)

            save_shards = False
            for file_chunks in tokenized_chunks:
                total_chunks += len(file_chunks)
                for triple in file_chunks:
                    if np.random.random() < 0.01:  # 1% chance for validation
                        [current_val_shard.extend(chunk) for chunk in triple]
                        # Honestly this should never happen since whenever a train shard is saved a val shard is, but...
                        # Just in case (/ leftover from old logic)
                        if len(current_val_shard) * 2 >= (SHARD_SIZE // 10):
                            save_shards = True
                    else:
                        [current_train_shard.extend(chunk) for chunk in triple]
                        if len(current_train_shard) * 2 >= SHARD_SIZE:
                            save_shards = True

            bar.set_description(
                f"Processing audio files (processed {total_chunks} chunks, avg. {total_chunks / (i + BATCH_SIZE):.1f}/file)")

            processed_count = i + BATCH_SIZE
            # Save state at least every 50 batches, and if we've saved any data.
            # Also ensures that any time train data is saved, val data is saved,
            # so the ratio of actually saved splits stays correct.
            if save_shards or (processed_count // BATCH_SIZE) % 50 == 0:
                save_shuffle_state(audio_files, processed_count)
                if current_val_shard:
                    val_shard_index = save_shard(current_val_shard, val_shard_index, 'val')
                    current_val_shard = []
                if current_train_shard:
                    train_shard_index = save_shard(current_train_shard, train_shard_index, 'train')
                    current_train_shard = []

    except KeyboardInterrupt:
        print("\nInterrupted. Saving progress...")
        save_shuffle_state(audio_files, processed_count)
        if current_val_shard:
            save_shard(current_val_shard, val_shard_index, 'val')
        if current_train_shard:
            save_shard(current_train_shard, train_shard_index, 'train')
        exit()


    # Save any remaining data in the last shards
    if current_train_shard:
        save_shard(current_train_shard, train_shard_index, 'train')
    if current_val_shard:
        save_shard(current_val_shard, val_shard_index, 'val')
    os.remove(state_file)


if __name__ == "__main__":
    main()

print("Done.")
