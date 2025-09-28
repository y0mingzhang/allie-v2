from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yimingzhang/lichess_tokens",
    repo_type="dataset",
    local_dir="data/tokens/",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "0001566-0001827/*",
        "0001827-0002088/*",
    ],
    resume_download=True,
)
