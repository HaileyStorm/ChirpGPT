import numpy as np
import os
from tqdm import tqdm

SEPARATOR_TOKEN = 4097


def analyze_dataset(data_root, split):
    shards = [s for s in os.listdir(data_root) if split in s]
    shards = sorted([os.path.join(data_root, s) for s in shards])

    for shard in tqdm(shards, desc=f"Analyzing {split} shards"):
        data = np.load(shard)
        print(f"\nAnalyzing shard: {os.path.basename(shard)}")
        print(f"Total tokens in shard: {len(data)}")

        # Print all 1578 tokens, 20 per line
        first_1578 = data[:1538]
        print(f"First 1578 tokens:")
        for i in range(0, len(first_1578), 20):
            chunk = first_1578[i:i + 20]
            print(' '.join(f'{token:5d}' for token in chunk))
        print("\n")

        # Find indices of separator tokens
        separator_indices = np.where(data == SEPARATOR_TOKEN)[0]
        print(f"Number of separator tokens: {len(separator_indices)}")

        if len(separator_indices) > 0:
            # Calculate distances between separators
            distances = np.diff(separator_indices)
            print(f"Min distance between separators: {np.min(distances)}")
            print(f"Max distance between separators: {np.max(distances)}")
            print(f"Mean distance between separators: {np.mean(distances):.2f}")
            print(f"Median distance between separators: {np.median(distances)}")

            # Print individual distances for closer inspection
            print("\nDistances between separators:")
            for i, distance in enumerate(distances[:20], 1):  # Print first 20 distances
                print(f"Distance {i}: {distance}")

            if len(distances) > 20:
                print("...")  # Indicate there are more distances

            # Check if any distance is not divisible by CRITICAL_DIVISOR (768)
            non_divisible = [d for d in distances if d % 768 != 0]
            if non_divisible:
                print(f"\nWARNING: Found {len(non_divisible)} distances not divisible by 768!")
                print(f"Problematic distances: {non_divisible[:10]}...")  # Show first 10

        print("\n" + "-" * 50 + "\n")  # Separator between shards

        # Optional: break after first shard to avoid too much output
        break  # Remove this line if you want to analyze all shards


def main():
    data_root = "./music_data"
    for split in ['train', 'val']:
        print(f"Analyzing {split} dataset...\n")
        analyze_dataset(data_root, split)


if __name__ == "__main__":
    main()