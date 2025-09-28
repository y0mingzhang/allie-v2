import os
import glob
from typing import List, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import numpy as np
from picotron.utils import print

import picotron.process_group_manager as pgm


class NpyTokenDataset(Dataset):
    """Virtually concatenates multiple 1D .npy token files and yields fixed-length windows.

    Windows do not cross file boundaries. If a single path is provided (no glob), it behaves as before.
    """

    def __init__(self, npy_glob_or_path: str, seq_length: int, num_samples: int | None = None):
        self.seq_length = seq_length

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
            if len(arr) < seq_length + 1:
                # Skip too-short files
                continue
            total_length = ((len(arr) - 1) // seq_length) * seq_length + 1
            num_seq = (total_length - 1) // seq_length
            if num_seq <= 0:
                continue
            self.files.append(arr)
            self.file_lengths.append(len(arr))
            self.file_num_sequences.append(num_seq)

        if not self.files:
            raise ValueError("All candidate .npy files are too short for the requested seq_length+1")

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

        start = local_idx * self.seq_length
        end = start + self.seq_length + 1
        seq = np.asarray(self.files[file_idx][start:end], dtype=np.int64)
        return {"input_ids": seq}


class MicroBatchDataLoader(DataLoader):
    def __init__(self, micro_batch_size, seq_length, npy_path, grad_acc_steps, device, num_workers, num_samples=None, pin_memory=True):
        self.micro_batch_size = micro_batch_size
        self.seq_length = seq_length
        self.grad_acc_steps = grad_acc_steps
        self.global_batch_size = micro_batch_size * grad_acc_steps * pgm.process_group_manager.dp_world_size
        self.num_global_micro_batches = self.global_batch_size // self.micro_batch_size

        self.seq_length_per_gpu = seq_length // pgm.process_group_manager.cp_world_size

        self.dataset = NpyTokenDataset(npy_glob_or_path=npy_path, seq_length=seq_length, num_samples=num_samples)

        self.sampler = DistributedSampler(
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