import numpy as np
import torch
import os
import random


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split,
                 ddp=False, master_process=False, seed=None, critical_divisor=1024):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.critical_divisor = critical_divisor
        assert split in {'train', 'val'}
        self.split = split
        assert T % critical_divisor == 0, f"T ({T}) must be divisible by critical_divisor ({critical_divisor})"

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        data_root = "./music_data_shuffled"
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
        self.block_order = np.arange(self.num_T_blocks)
        #np.random.shuffle(self.block_order)

    def get_block(self, block_index):
        start = block_index * self.T
        end = start + self.T
        return self.tokens[start:end]

    def skip_batches(self, num_batches):
        print(f"Resume: skipping {num_batches} batches in {self.split} dataloader...")
        for _ in range(num_batches):
            self.next_batch(False, skip=True)

    def next_batch(self, loss_by_later_subchunks, skip=False):
        if self.current_batch >= self.num_batches:
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.load_and_shuffle_shard()
            self.current_batch = 0
        if skip:
            self.current_batch += 1
            return None

        B, T = self.B, self.T

        # Select B * num_processes blocks for this batch
        start_block = self.current_batch * B * self.num_processes
        batch_blocks = self.block_order[start_block: start_block + B * self.num_processes]

        # Select the blocks for this process
        process_blocks = batch_blocks[self.process_rank::self.num_processes]

        # Return full sequence & next-token offset full sequence, or full sequence & offset broken-down 2nd&3rd subchunks
        if loss_by_later_subchunks:
            x = torch.stack([self.get_block(i) for i in process_blocks])
            y = torch.stack(
                [self.get_block(i)[self.critical_divisor+1:2 * self.critical_divisor] for i in process_blocks])
            separator_token = torch.full((y.size(0), 1), 4097, dtype=y.dtype, device=y.device)
            y = torch.cat([y, separator_token], dim=1)
            z = torch.stack([self.get_block(i)[2 * self.critical_divisor+1:] for i in process_blocks])
            z = torch.cat([z, separator_token], dim=1)
            self.current_batch += 1
            return x, y, z
        else:
            x = torch.stack([self.get_block(i) for i in process_blocks])
            y = torch.stack([self.get_block(i)[1:] for i in process_blocks])
            separator_token = torch.full((y.size(0), 1), 4097, dtype=y.dtype, device=y.device)
            y = torch.cat([y, separator_token], dim=1)
            self.current_batch += 1
            return x, y

    def __len__(self):
        total_critical_blocks = sum(len(self.load_tokens(shard)) // self.critical_divisor for shard in self.shards)
        total_T_blocks = total_critical_blocks * self.critical_divisor // self.T
        return total_T_blocks // (self.B * self.num_processes)
