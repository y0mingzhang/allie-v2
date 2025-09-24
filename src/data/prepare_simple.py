#!/usr/bin/env python3

import glob
import random

from megatron.core.datasets import indexed_dataset
import numpy as np
from tqdm.auto import tqdm


def main():
    # Load all .npy files from data/tokens/
    npy_files = glob.glob("data/tokens/*.npy")
    print(f"Found {len(npy_files)} token files")

    # Load all documents
    documents = []
    total_tokens = 0
    for file_path in tqdm(npy_files):
        tokens = np.load(file_path)
        documents.append(tokens)
        total_tokens += len(tokens)

    # Shuffle documents
    random.seed(102943)
    random.shuffle(documents)

    output_bin = "data/bin/0923_processed_data.bin"
    output_idx = "data/bin/0923_processed_data.idx"

    print(
        f"Writing {len(documents)} documents ({total_tokens} tokens) to {output_bin} and {output_idx}..."
    )
    builder = indexed_dataset.IndexedDatasetBuilder(output_bin, dtype=np.uint16)

    # Add each document separately
    for doc_tokens in documents:
        builder.add_document(doc_tokens, [len(doc_tokens)])

    builder.finalize(output_idx)
    print("Done!")


if __name__ == "__main__":
    main()
