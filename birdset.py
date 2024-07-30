import os
import io
import torch
import torchaudio
import numpy as np
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from speech_tokenizer import SpeechTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = '/media/hailey/More/AI/gpt2audio/birdset_data_trainOnly_widerNet'
PREFIX = 'birds_wider'
SHARD_SIZE = 5 * 1024 * 1024  # 5MB in bytes
CHUNK_LENGTH = 9  # seconds
SUB_CHUNK_LENGTH = 4.5  # seconds

LAT_MIN = 27  # 33
LAT_MAX = 69  # 51
LONG_MIN = -162  # -125
LONG_MAX = -52  # -85

# Dataset configuration
DATASETS = {
    'HSN': ['train'],  #, 'test'],
    'POW': ['train'],  #, 'test'],  # 100% will fail original lat/long filter
    'SSW': ['train'],  #, 'test'],  # 100% will fail original lat/long filter
    'SNE': ['train'],  #, 'test']
    'XCL': ['train'],
}

# Initialize tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = SpeechTokenizer(device=device)


def filter_example(example, is_train_split):
    latitude = example['lat']
    longitude = example['long']
    length = example['length']
    if latitude is None or longitude is None:
        return False
    if length is None:
        # chunk_audio will handle it if it turns out to be shorter
        length = CHUNK_LENGTH

    base_condition = (
            LAT_MIN <= latitude <= LAT_MAX and
            LONG_MIN <= longitude <= LONG_MAX and
            length >= CHUNK_LENGTH
    )

    if is_train_split:
        quality = example['quality']
        num_events = len(example['detected_events'])
        if quality is None or num_events is None:
            return False
        return base_condition and (quality == 'A' or quality == 'B') and num_events >= length // CHUNK_LENGTH
    else:
        return base_condition


def chunk_audio(waveform, sample_rate, is_train_split, detected_events=None, start_time=None, end_time=None):
    if not is_train_split:
        # For test splits, create a single CHUNK_LENGTH-second chunk centered around the vocalization
        center = (start_time + end_time) / 2
        chunk_start = max(0, center - (CHUNK_LENGTH / 2))
        chunk_end = min(waveform.shape[1] / sample_rate, chunk_start + CHUNK_LENGTH)
        chunk_start = max(0, chunk_end - CHUNK_LENGTH)  # Ensure full CHUNK_LENGTH
        chunk = waveform[:, int(chunk_start * sample_rate):int(chunk_end * sample_rate)]
        #logger.info(f"Test split: Created chunk from {chunk_start:.2f}s to {chunk_end:.2f}s")
        return [chunk]

    # For train splits
    audio_length = waveform.shape[1] / sample_rate
    chunks = []
    previous_chunk_end = 0

    def add_chunk(start, end):
        if end - start >= CHUNK_LENGTH and end <= audio_length:
            chunk = waveform[:, int(start * sample_rate):int(end * sample_rate)]
            chunks.append(chunk)
            #logger.info(f"Added chunk: {start:.2f}s to {end:.2f}s")
            return end
        else:
            #logger.info(f"Skipped short or out-of-bounds chunk: {start:.2f}s to {end:.2f}s, file end {audio_length}")
            return start

    i = 0
    while i < len(detected_events):
        event_start, event_end = detected_events[i]
        event_duration = event_end - event_start

        if event_duration >= CHUNK_LENGTH:
            # Handle events longer than CHUNK_LENGTH seconds
            chunk_start = max(event_start, previous_chunk_end)
            while chunk_start < event_end:
                chunk_end = chunk_start + CHUNK_LENGTH
                # Check if we can include the next full event(s)
                next_event_index = i + 1
                while next_event_index < len(detected_events) and detected_events[next_event_index][1] <= chunk_end:
                    chunk_end = detected_events[next_event_index][0]
                    next_event_index += 1
                previous_chunk_end = add_chunk(chunk_start, chunk_end)
                if previous_chunk_end == chunk_start:
                    break
                chunk_start = previous_chunk_end
            i = next_event_index - 1 if next_event_index > i + 1 else i
        else:
            # Try to create a chunk with multiple events
            chunk_start = max(event_start, previous_chunk_end)
            chunk_end = min(audio_length, chunk_start + CHUNK_LENGTH)
            next_event_index = i + 1

            # Include as many events as possible within CHUNK_LENGTH seconds
            while next_event_index < len(detected_events) and detected_events[next_event_index][1] <= chunk_end:
                next_event_index += 1

            # Adjust chunk_end to include the last partial event if possible
            if next_event_index < len(detected_events) and detected_events[next_event_index][0] < chunk_end:
                if detected_events[next_event_index][1] - chunk_start <= CHUNK_LENGTH:
                    chunk_end = min(audio_length, detected_events[next_event_index][1])
                    next_event_index += 1

            previous_chunk_end = add_chunk(chunk_start, chunk_end)
            i = next_event_index - 1

        i += 1

    #logger.info(f"Total chunks created: {len(chunks)}")
    return chunks


def create_chunks(waveform, sample_rate, chunk_length):
    total_samples = waveform.shape[1]
    samples_per_chunk = int(chunk_length * sample_rate)
    num_chunks = total_samples // samples_per_chunk

    chunks = []
    for i in range(num_chunks):
        start = i * samples_per_chunk
        end = (i + 1) * samples_per_chunk
        chunk = waveform[:, start:end]
        chunks.append(chunk)

    return chunks


def process_audio(audio_bytes, is_train_split, detected_events=None, start_time=None, end_time=None):
    audio_io = io.BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(audio_io, format="ogg")

    # Resample if necessary
    if sample_rate != tokenizer.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=tokenizer.sample_rate)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    chunks = chunk_audio(waveform, tokenizer.sample_rate, is_train_split, detected_events, start_time, end_time)

    tokenized_chunks = []
    for chunk in chunks:
        sub_chunks = create_chunks(chunk, tokenizer.sample_rate, SUB_CHUNK_LENGTH)
        for sub_chunk in sub_chunks:
            tokenized_sub_chunk = tokenizer.encode([sub_chunk])
            tokenized_chunks.append(tokenized_sub_chunk[0][:-1])

    return tokenized_chunks


def get_next_shard_index(shard_type):
    existing_shards = [f for f in os.listdir(DATA_DIR) if f.startswith(f'{PREFIX}_{shard_type}_') and f.endswith('.npy')]
    if not existing_shards:
        return 0
    return max([int(f.split('_')[-1].split('.')[0]) for f in existing_shards]) + 1


def main(resume_index=0, datasets_to_use=DATASETS):
    os.makedirs(DATA_DIR, exist_ok=True)

    train_shard_index = get_next_shard_index('train')
    val_shard_index = get_next_shard_index('val')
    current_train_shard = []
    current_val_shard = []
    current_train_shard_size = 0
    current_val_shard_size = 0

    def save_shard(shard, shard_size, shard_type):
        nonlocal train_shard_index, val_shard_index
        if shard:
            shard_path = os.path.join(DATA_DIR,
                                      f'{PREFIX}_{shard_type}_{train_shard_index if shard_type == "train" else val_shard_index:04d}.npy')
            np.save(shard_path, np.array(shard, dtype=np.int16))
            print(f"\nSaved {shard_type} shard: {shard_path}")
            if shard_type == 'train':
                train_shard_index += 1
            else:
                val_shard_index += 1
            return [], 0
        return shard, shard_size

    for subset, splits in datasets_to_use.items():
        for split in splits:
            dataset = load_dataset('DBD-research-group/BirdSet', subset, split=split, streaming=True, trust_remote_code=True)
            is_train_split = split == 'train'

            for i, example in tqdm(enumerate(dataset), desc=f"Processing {subset} {split}"):
                if i < resume_index:
                    continue
                resume_index = 0

                if filter_example(example, is_train_split):
                    detected_events = example.get('detected_events')
                    start_time = example.get('start_time')
                    end_time = example.get('end_time')

                    tokenized_chunks = process_audio(
                        example['audio']['bytes'],
                        is_train_split,
                        detected_events,
                        start_time,
                        end_time
                    )

                    for chunk in tokenized_chunks:
                        if np.random.random() < 0.01:  # 1% chance for validation
                            current_val_shard.extend(chunk)
                            current_val_shard_size += len(chunk) * 2
                            if current_val_shard_size >= SHARD_SIZE:
                                current_val_shard, current_val_shard_size = save_shard(current_val_shard,
                                                                                       current_val_shard_size, 'val')
                        else:
                            current_train_shard.extend(chunk)
                            current_train_shard_size += len(chunk) * 2
                            if current_train_shard_size >= SHARD_SIZE:
                                current_train_shard, current_train_shard_size = save_shard(current_train_shard,
                                                                                           current_train_shard_size,
                                                                                           'train')
                    if i % 10000 == 0:
                        current_train_shard, current_train_shard_size = save_shard(current_train_shard,
                                                                                   current_train_shard_size, 'train')
                        current_val_shard, current_val_shard_size = save_shard(current_val_shard,
                                                                               current_val_shard_size, 'val')

            # Save any remaining data for this in the last shard
            current_train_shard, current_train_shard_size = save_shard(current_train_shard, current_train_shard_size, 'train')
            current_val_shard, current_val_shard_size = save_shard(current_val_shard, current_val_shard_size, 'val')


if __name__ == "__main__":
    # Example: Exclude XCL, POW, and SSW
    datasets_to_use = {k: v for k, v in DATASETS.items()}  #if k not in ['POW', 'SSW', 'XCL']}
    main(resume_index=0, datasets_to_use=datasets_to_use)
