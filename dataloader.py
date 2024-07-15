import numpy as np
import torch
import os
import random


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split,
                 ddp=False, master_process=False, seed=None, critical_divisor=512):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.critical_divisor = critical_divisor
        assert split in {'train', 'val'}
        assert T % critical_divisor == 0, f"T ({T}) must be divisible by critical_divisor ({critical_divisor})"

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        data_root = "./birdset_data_trainOnly_shuffled"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]

        random.shuffle(shards)

        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"

        self.total_tokens = 0
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
            for shard in self.shards:
                npt = np.load(shard, mmap_mode='r')
                self.total_tokens += len(npt)
            print(f"Total tokens across all shards: {self.total_tokens}")
        self.total_tokens = torch.tensor([self.total_tokens], dtype=torch.long)
        if ddp:
            torch.distributed.broadcast(self.total_tokens, src=0)
        self.total_tokens = self.total_tokens.item()

        self.reset()

    @staticmethod
    def load_tokens(filename):
        npt = np.load(filename, mmap_mode='r')
        if npt.dtype != np.int64:
            npt = npt.astype(np.int64)
        return torch.from_numpy(npt)

    def reset(self):
        self.current_shard = 0
        self.load_and_shuffle_shard()
        self.current_batch = 0

    def load_and_shuffle_shard(self):
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        assert len(self.tokens) % self.critical_divisor == 0, \
            f"Shard size {len(self.tokens)} is not divisible by critical_divisor={self.critical_divisor}"

        # Calculate the number of critical blocks in this shard
        self.num_critical_blocks = len(self.tokens) // self.critical_divisor

        # Calculate the number of complete T-sized blocks we can form
        self.num_T_blocks = self.num_critical_blocks * self.critical_divisor // self.T

        # Calculate the number of complete batches in this shard
        self.num_batches = self.num_T_blocks // (self.B * self.num_processes)

        # Create shuffled indices for all critical blocks in the shard
        self.block_order = np.arange(self.num_critical_blocks)
        np.random.shuffle(self.block_order)

    def get_block(self, block_index):
        start = block_index * self.critical_divisor
        end = start + self.T
        if end <= len(self.tokens):
            return self.tokens[start:end]
        else:
            # Wrap to the next shard
            next_shard = (self.current_shard + 1) % len(self.shards)
            next_tokens = self.load_tokens(self.shards[next_shard])
            return torch.cat([self.tokens[start:], next_tokens[:end - len(self.tokens)]])

    def next_batch(self):
        if self.current_batch >= self.num_batches:
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.load_and_shuffle_shard()
            self.current_batch = 0

        B, T = self.B, self.T

        # Select B * num_processes blocks for this batch
        start_block = self.current_batch * B * self.num_processes
        batch_blocks = self.block_order[start_block: start_block + B * self.num_processes]

        # Select the blocks for this process
        process_blocks = batch_blocks[self.process_rank::self.num_processes]

        # Gather the tokens for these blocks
        x = torch.stack([self.get_block(i) for i in process_blocks])

        # For y, we need to handle the case where we're at the last block
        y_list = []
        for i in process_blocks:
            current_block = self.get_block(i)
            if i < self.num_critical_blocks - 1:
                next_block = self.get_block(i + 1)
                y_list.append(torch.cat([current_block[1:], next_block[:1]]))
            else:
                # If it's the last block, wrap to the first block of the next shard
                next_block = self.get_block(0)  # This will automatically load from the next shard
                y_list.append(torch.cat([current_block[1:], next_block[:1]]))

        y = torch.stack(y_list)

        self.current_batch += 1
        return x, y

    def __len__(self):
        total_critical_blocks = sum(len(self.load_tokens(shard)) // self.critical_divisor for shard in self.shards)
        total_T_blocks = total_critical_blocks * self.critical_divisor // self.T
        return total_T_blocks // (self.B * self.num_processes)