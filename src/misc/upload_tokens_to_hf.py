#

from huggingface_hub import upload_large_folder

upload_large_folder(
    repo_id="yimingzhang/lichess_tokens_v2",
    folder_path="data/tokens_v2/",
    repo_type="dataset",
    num_workers=16,
)
