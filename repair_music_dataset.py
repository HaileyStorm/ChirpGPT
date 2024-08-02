import numpy as np
import os
from tqdm import tqdm

SEPARATOR_TOKEN = 4097
BLOCK_SIZE = 768
PROBLEM_SIZE = 640


def repair_shard(data):
    result = []
    discarded_chunks = 0
    total_chunks = len(data) // (BLOCK_SIZE * 2)
    i = 0
    while i < len(data):
        if (i + BLOCK_SIZE * 2 <= len(data)) and \
                (data[i] == SEPARATOR_TOKEN) and \
                (data[i + BLOCK_SIZE] == SEPARATOR_TOKEN) and \
                (i + BLOCK_SIZE * 2 == len(data) or data[i + BLOCK_SIZE * 2] == SEPARATOR_TOKEN):
            # Good chunk pair
            result.extend(data[i:i + BLOCK_SIZE * 2])
            i += BLOCK_SIZE * 2
        elif ((i + BLOCK_SIZE + PROBLEM_SIZE <= len(data)) and
              (data[i] == SEPARATOR_TOKEN) and
              (data[i + BLOCK_SIZE] == SEPARATOR_TOKEN) and
              ((i + BLOCK_SIZE + PROBLEM_SIZE == len(data)) or
               (i + BLOCK_SIZE + PROBLEM_SIZE < len(data) and data[
                   i + BLOCK_SIZE + PROBLEM_SIZE] == SEPARATOR_TOKEN))):
            # Bad chunk detected, discard entire pair
            #print(f"Found bad chunk at position {i}, discarding 768+640 tokens")
            discarded_chunks += 1
            i += BLOCK_SIZE + PROBLEM_SIZE
        else:
            # If we're near the end of the data, just stop
            if len(data) - i < BLOCK_SIZE * 2:
                break
            raise ValueError(f"Encountered unexpected token sequence at position: {i}")

    #print(f"Total chunks: {total_chunks}, Discarded chunks: {discarded_chunks}")
    #print(f"Percentage discarded: {discarded_chunks / total_chunks * 100:.2f}%")

    return np.array(result, dtype=np.int16), discarded_chunks, total_chunks


def repair_dataset(input_dir, output_dir, split):
    os.makedirs(output_dir, exist_ok=True)
    shards = [s for s in os.listdir(input_dir) if split in s and s.endswith('.npy')]
    total_discarded, grand_total = 0, 0

    for idx, shard_name in enumerate(tqdm(shards, desc=f"Repairing {split} shards")):
        input_path = os.path.join(input_dir, shard_name)
        data = np.load(input_path)

        try:
            repaired_data, discarded, total = repair_shard(data)
            total_discarded += discarded
            grand_total += total
            if len(repaired_data) % BLOCK_SIZE != 0:
                raise ValueError(f"Repaired shard {shard_name} not multiple of {BLOCK_SIZE}: {len(repaired_data)}")

            output_path = os.path.join(output_dir, f"{split}_repaired_{idx:03d}.npy")
            np.save(output_path, repaired_data)

        except ValueError as e:
            print(f"Failed to repair {shard_name}: {e}")

    print(f"Overall for {split}: discarded {total_discarded}/{grand_total} chunks ({total_discarded / grand_total * 100:.2f}%)")


def main():
    DATA_DIR = "./music_data"  # Your input dir
    OUTPUT_DIR = "./repaired_music_data"  # Your output dir

    for split in ['train', 'val']:
        repair_dataset(DATA_DIR, OUTPUT_DIR, split)


if __name__ == "__main__":
    main()
