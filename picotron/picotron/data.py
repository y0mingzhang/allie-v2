import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import numpy as np
from picotron.utils import print

import picotron.process_group_manager as pgm


class NpyTokenDataset(Dataset):
    def __init__(self, npy_path, seq_length, num_samples=None):
        self.seq_length = seq_length
        self.tokens = np.load(npy_path, mmap_mode='r')
        if self.tokens.ndim != 1:
            raise ValueError("Expected a 1D array of token ids in the .npy file")
        if len(self.tokens) < seq_length + 1:
            raise ValueError("Token array too small for the requested seq_length+1")

        total_length = ((len(self.tokens) - 1) // seq_length) * seq_length + 1
        self.num_sequences = (total_length - 1) // seq_length
        if num_samples is not None:
            self.num_sequences = min(self.num_sequences, int(num_samples))

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length + 1
        seq = np.asarray(self.tokens[start:end], dtype=np.int64)
        return {"input_ids": seq}


class MicroBatchDataLoader(DataLoader):
    def __init__(self, micro_batch_size, seq_length, npy_path, grad_acc_steps, device, num_workers, num_samples=None, pin_memory=True):
        self.micro_batch_size = micro_batch_size
        self.seq_length = seq_length
        self.grad_acc_steps = grad_acc_steps
        self.global_batch_size = micro_batch_size * grad_acc_steps * pgm.process_group_manager.dp_world_size
        self.num_global_micro_batches = self.global_batch_size // self.micro_batch_size

        self.seq_length_per_gpu = seq_length // pgm.process_group_manager.cp_world_size

        self.dataset = NpyTokenDataset(npy_path=npy_path, seq_length=seq_length, num_samples=num_samples)

        self.sampler = DistributedSampler(
            self.dataset,
            num_replicas=pgm.process_group_manager.dp_world_size,
            rank=pgm.process_group_manager.dp_rank,
            shuffle=False,
        )

        super().__init__(
            self.dataset,
            batch_size=micro_batch_size,
            collate_fn=self.collate_batch,
            pin_memory=pin_memory,
            num_workers=num_workers,
            sampler=self.sampler,
            shuffle=False,
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