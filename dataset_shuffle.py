import numpy as np
import os
import random
from tqdm import tqdm

# Constants
BLOCK_SIZE = 1024
CRITICAL_DIVISOR = 512
TARGET_SHARD_BLOCKS = 1230


def load_dataset(data_root, split):
    shards = [s for s in os.listdir(data_root) if split in s]
    shards = sorted([os.path.join(data_root, s) for s in shards])

    all_data = []
    for shard in tqdm(shards, desc=f"Loading {split} shards"):
        data = np.load(shard)
        if len(data) % CRITICAL_DIVISOR != 0:
            raise ValueError(f"Shard {shard} length is not divisible by {CRITICAL_DIVISOR}")
        all_data.append(data)

    return all_data


def process_shard(shard_data):
    full_blocks = len(shard_data) // BLOCK_SIZE
    remainder = len(shard_data) % BLOCK_SIZE

    processed_blocks = []
    for i in range(full_blocks):
        start = i * BLOCK_SIZE
        end = (i + 1) * BLOCK_SIZE
        processed_blocks.append(shard_data[start:end])

    if remainder > 0:
        if remainder % CRITICAL_DIVISOR == 0:
            # Handle the last blocks that are of size exactly a multiple of CRITICAL_DIVISOR
            while remainder:
                processed_blocks.append(shard_data[-remainder:-(remainder - CRITICAL_DIVISOR)])
                remainder -= CRITICAL_DIVISOR
        else:
            raise ValueError("Shard not divisible by CRITICAL_DIVISOR!")

    return processed_blocks

def shuffle_and_create_new_shards(all_data, output_dir, split):
    os.makedirs(output_dir, exist_ok=True)

    all_blocks = []
    for shard_data in all_data:
        all_blocks.extend(process_shard(shard_data))

    # Combine blocks of CRITICAL_DIVISOR size if possible
    critical_blocks = [block for block in all_blocks if len(block) == CRITICAL_DIVISOR]
    other_blocks = [block for block in all_blocks if len(block) == BLOCK_SIZE]

    # Combine CRITICAL_DIVISOR blocks into BLOCK_SIZE blocks where possible
    combined_blocks = []
    while len(critical_blocks) >= (BLOCK_SIZE // CRITICAL_DIVISOR):
        combined_block = np.concatenate(critical_blocks[:BLOCK_SIZE // CRITICAL_DIVISOR])
        combined_blocks.append(combined_block)
        critical_blocks = critical_blocks[BLOCK_SIZE // CRITICAL_DIVISOR:]

    # Any remaining CRITICAL_DIVISOR blocks that cannot be combined are added as they are
    all_blocks = other_blocks + combined_blocks + critical_blocks
    random.shuffle(all_blocks)

    total_blocks = len(all_blocks)
    num_shards = (total_blocks + TARGET_SHARD_BLOCKS - 1) // TARGET_SHARD_BLOCKS

    for i in tqdm(range(num_shards), desc=f"Creating new {split} shards"):
        start = i * TARGET_SHARD_BLOCKS
        end = min((i + 1) * TARGET_SHARD_BLOCKS, total_blocks)
        shard_blocks = all_blocks[start:end]

        shard_data = np.concatenate(shard_blocks)
        filename = os.path.join(output_dir, f"{split}_shard_{i:03d}.npy")
        np.save(filename, shard_data)



def main():
    data_root = "./birdset_data_trainOnly"
    output_root = "./birdset_data_trainOnly_shuffled"

    for split in ['train', 'val']:
        print(f"Processing {split} dataset...")
        all_data = load_dataset(data_root, split)
        shuffle_and_create_new_shards(all_data, output_root, split)

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()