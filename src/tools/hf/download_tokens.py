from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yimingzhang/lichess_tokens_v2",
    repo_type="dataset",
    local_dir="data/tokens_v2/",
    local_dir_use_symlinks=False,
    resume_download=True,
)
