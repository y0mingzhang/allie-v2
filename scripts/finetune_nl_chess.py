#!/usr/bin/env python
"""Fine-tune Qwen3-0.6B on natural language chess games with SF commentary.

Uses HF Trainer for simplicity. The model learns to predict both moves and
evaluations, leveraging its pretrained English understanding.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/finetune_nl_chess.py \
        --data ~/user_data/chess-v3/data/nl_chess/train.jsonl \
        --model Qwen/Qwen3-0.6B \
        --output-dir ~/user_data/chess-v3/models/nl_chess_qwen3 \
        --epochs 1 --lr 2e-5
"""

import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


class NLChessDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path) as f:
            for line in f:
                data = json.loads(line)
                tokens = tokenizer(
                    data["text"],
                    truncation=True,
                    max_length=max_length,
                    return_tensors=None,
                )
                if len(tokens["input_ids"]) >= 50:  # skip very short
                    self.examples.append(tokens)

        print(f"Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k: v for k, v in self.examples[idx].items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="models/nl_chess")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    print(f"\nLoading data: {args.data}")
    dataset = NLChessDataset(args.data, tokenizer, args.max_length)

    # Split 95/5 train/eval
    n_eval = max(50, len(dataset) // 20)
    n_train = len(dataset) - n_eval
    train_ds, eval_ds = torch.utils.data.random_split(
        dataset, [n_train, n_eval], generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {n_train}, Eval: {n_eval}")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        report_to="wandb",
        run_name="nl-chess-qwen3-0.6B",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
