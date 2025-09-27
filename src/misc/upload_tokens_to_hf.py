#

from huggingface_hub import upload_large_folder

upload_large_folder(
    repo_id="yimingzhang/lichess_tokens",
    folder_path="data/tokens/",
    repo_type="dataset",
    num_workers=8,
)
