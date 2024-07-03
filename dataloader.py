import numpy as np
import torch
import os

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, ddp=False, master_process=False):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "./data"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"

        self.total_tokens = 0
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
            # Count total tokens across all shards
            for shard in self.shards:
                # Use numpy's memmap for memory efficiency
                npt = np.load(shard, mmap_mode='r')
                self.total_tokens += len(npt)
            print(f"Total tokens across all shards: {self.total_tokens}")
        # Broadcast total_tokens to all processes
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
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y