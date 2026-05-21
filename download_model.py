from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="GAASH-Lab/Grounded-Teacher",
    repo_type="model",          # important
    local_dir="download",
    local_dir_use_symlinks=False
)

print("Downloaded to:", local_dir)
