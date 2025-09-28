from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yimingzhang/lichess_tokens",
    repo_type="dataset",
    local_dir="data/tokens/",
    local_dir_use_symlinks=False,
    resume_download=True,
)
