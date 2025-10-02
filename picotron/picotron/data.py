import os
import glob
from typing import List, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import numpy as np

import picotron.process_group_manager as pgm


class NpyTokenDataset(Dataset):
    """Reads pre-packed batches from .npy token files.

    Assumes the .npy files contain pre-packed batches where each batch is (seq_length + 1) tokens,
    and each batch starts with a new game (BOS token).
    """

    def __init__(self, npy_glob_or_path: str, seq_length: int, num_samples: int | None = None):
        self.seq_length = seq_length
        self.pack_size = seq_length + 1  # Each batch is seq_length + 1 tokens

        # Resolve paths from glob or single path
        paths: List[str]
        if any(ch in npy_glob_or_path for ch in ["*", "?", "["]):
            paths = sorted(glob.glob(npy_glob_or_path))
        elif os.path.isdir(npy_glob_or_path):
            paths = sorted(glob.glob(os.path.join(npy_glob_or_path, "*.npy")))
        else:
            paths = [npy_glob_or_path]

        if not paths:
            raise ValueError(f"No .npy files found for pattern/path: {npy_glob_or_path}")

        self.files: List[np.memmap] = []
        self.file_lengths: List[int] = []
        self.file_num_sequences: List[int] = []
        for p in paths:
            arr = np.load(p, mmap_mode='r')
            if arr.ndim != 1:
                raise ValueError(f"Expected 1D array in {p}, got shape {arr.shape}")
            if len(arr) < self.pack_size:
                # Skip too-short files
                continue
            # Each sequence is exactly pack_size tokens (seq_length + 1)
            num_seq = len(arr) // self.pack_size
            if num_seq <= 0:
                continue
            self.files.append(arr)
            self.file_lengths.append(len(arr))
            self.file_num_sequences.append(num_seq)

        if not self.files:
            raise ValueError(f"All candidate .npy files are too short for pack_size={self.pack_size}")

        # Prefix sum to map global idx -> (file_idx, local_idx)
        self.file_index_offsets: List[int] = [0]
        s = 0
        for n in self.file_num_sequences:
            s += n
            self.file_index_offsets.append(s)

        total_sequences = self.file_index_offsets[-1]
        if num_samples is not None:
            total_sequences = min(total_sequences, int(num_samples))
        self.num_sequences = total_sequences

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(idx)

        # Binary search over file_index_offsets to find file containing this idx
        # offsets: [0, n0, n0+n1, ..., total]
        left, right = 0, len(self.file_index_offsets) - 1
        while left < right:
            mid = (left + right) // 2
            if self.file_index_offsets[mid + 1] <= idx:
                left = mid + 1
            elif self.file_index_offsets[mid] > idx:
                right = mid - 1
            else:
                right = mid
                break
        file_idx = left if self.file_index_offsets[left] <= idx < self.file_index_offsets[left + 1] else right
        local_idx = idx - self.file_index_offsets[file_idx]

        # Read a fixed-size batch (seq_length + 1 tokens)
        start = local_idx * self.pack_size
        end = start + self.pack_size
        seq = np.asarray(self.files[file_idx][start:end], dtype=np.int64)
        return {"input_ids": seq}


class SkippableDistributedSampler(DistributedSampler):
    """DistributedSampler with efficient skipping capability for fast-forward resume.

    Allows skipping ahead to a specific sample position without iterating through all samples.
    The skip is applied once on the next __iter__() call, then automatically reset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_samples = 0

    def set_skip(self, num_samples: int):
        """Set the number of samples to skip on the next __iter__() call."""
        self.skip_samples = num_samples

    def __iter__(self):
        # Generate indices using parent class logic
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * ((padding_size + len(indices) - 1) // len(indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[:self.total_size]

        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]

        # Apply skip if set, then reset it
        if self.skip_samples > 0:
            indices = indices[self.skip_samples:]
            self.skip_samples = 0

        return iter(indices)


class MicroBatchDataLoader(DataLoader):
    def __init__(self, micro_batch_size, seq_length, npy_path, grad_acc_steps, device, num_workers, num_samples=None, pin_memory=True):
        self.micro_batch_size = micro_batch_size
        self.seq_length = seq_length
        self.grad_acc_steps = grad_acc_steps
        self.global_batch_size = micro_batch_size * grad_acc_steps * pgm.process_group_manager.dp_world_size
        self.num_global_micro_batches = self.global_batch_size // self.micro_batch_size

        self.seq_length_per_gpu = seq_length // pgm.process_group_manager.cp_world_size

        self.dataset = NpyTokenDataset(npy_glob_or_path=npy_path, seq_length=seq_length, num_samples=num_samples)

        self.sampler = SkippableDistributedSampler(
            self.dataset,
            num_replicas=pgm.process_group_manager.dp_world_size,
            rank=pgm.process_group_manager.dp_rank,
            shuffle=True,
            seed=3249876,
        )

        super().__init__(
            self.dataset,
            batch_size=micro_batch_size,
            collate_fn=self.collate_batch,
            pin_memory=pin_memory,
            num_workers=num_workers,
            sampler=self.sampler,
        )

    def collate_batch(self, batch):
        batch_input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
        batch_size = batch_input_ids.size(0)
        start_idx = pgm.process_group_manager.cp_rank * self.seq_length_per_gpu
        end_idx = start_idx + self.seq_length_per_gpu
        input_ids = batch_input_ids[:, start_idx:end_idx].contiguous()
        target_ids = batch_input_ids[:, start_idx+1:end_idx+1].contiguous()
        position_ids = torch.arange(start_idx, end_idx, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "position_ids": position_ids,
            "hidden_states": None,
        }

    def fast_forward_to_step(self, target_step):
        """
        Fast-forward the dataloader to a specific training step.

        Args:
            target_step (int): The training step to fast-forward to
        """
        # Each training step consumes grad_acc_steps micro-batches
        batches_to_skip = target_step * self.grad_acc_steps

        # Calculate which epoch we should be in and how many batches to skip within that epoch
        # Use batch-based calculation to properly account for DistributedSampler padding
        # Use ceil to account for partial batches (padding makes epochs evenly divisible)
        import math
        batches_per_epoch = math.ceil(len(self.sampler) / self.micro_batch_size)
        epoch_offset = batches_to_skip // batches_per_epoch
        batches_within_epoch = batches_to_skip % batches_per_epoch

        # Convert batches to samples for the skip operation
        samples_within_epoch = batches_within_epoch * self.micro_batch_size

        # Set the epoch (this affects the RNG seed for shuffling)
        self.sampler.set_epoch(epoch_offset)

        # Set the skip position (will be applied on next __iter__ call)
        # Note: set_skip expects number of SAMPLES to skip in the sampler's index space
        self.sampler.set_skip(samples_within_epoch)

        # Reset the iterator so it picks up the new epoch and skip settings
        self._iterator = None

    def __iter__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.sampler.set_epoch(self.sampler.epoch + 1 if hasattr(self.sampler, 'epoch') else 0)
            self._iterator = super().__iter__()
            try:
                batch = next(self._iterator)
            except StopIteration:
                self._iterator = None
                raise StopIteration
        return batch