#

from huggingface_hub import upload_large_folder

upload_large_folder(
    repo_id="yimingzhang/lichess_tokenized",
    folder_path="data/dataset_v2/",
    repo_type="dataset",
    num_workers=8,
)
