from huggingface_hub import upload_large_folder

upload_large_folder(
    repo_id="yimingzhang/lichess_tokenized",
    folder_path="data/dataset/",
    repo_type="dataset",
    num_workers=24,
)
