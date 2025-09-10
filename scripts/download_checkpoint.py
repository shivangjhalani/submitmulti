import os
import sys
from typing import Optional

from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError, GatedRepoError


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value is not None else default


def try_download(repo_id: str, repo_type: str, local_dir: str, allow_pattern: str) -> bool:
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=[allow_pattern],
            token=None,  # Force anonymous download (no token)
        )
        return True
    except (RepositoryNotFoundError, GatedRepoError, HfHubHTTPError):
        return False


def main() -> int:
    repo_id = get_env("CHECKPOINT_REPO_ID", "ThefirstM/checkpoints")
    allow_pattern = get_env("CHECKPOINT_PATTERN", "aokvqa_cot_aokvqa-cot-stage0/epoch-8/**")
    local_dir = get_env("CHECKPOINT_LOCAL_DIR", "checkpoints")

    # Anonymous download only (no token)
    repo_type_env = (os.getenv("CHECKPOINT_REPO_TYPE") or "").strip().lower()

    os.makedirs(local_dir, exist_ok=True)

    # Skip download if already present
    expected_dir = os.path.join(local_dir, allow_pattern.split("/**")[0])
    if os.path.isdir(expected_dir) and any(True for _ in os.scandir(expected_dir)):
        print(f"Checkpoint already present at: {expected_dir}")
        return 0

    # Try as forced repo_type, else try dataset then model
    if repo_type_env in {"dataset", "model"} and try_download(repo_id, repo_type_env, local_dir, allow_pattern):
        print(f"Downloaded from {repo_type_env} repo: {repo_id}")
        return 0

    if try_download(repo_id, "dataset", local_dir, allow_pattern):
        print(f"Downloaded from dataset repo: {repo_id}")
        return 0
    if try_download(repo_id, "model", local_dir, allow_pattern):
        print(f"Downloaded from model repo: {repo_id}")
        return 0

    print(
        (
            "Failed to download checkpoint anonymously. Ensure the repo is public or "
            "pre-download and mount 'checkpoints/aokvqa_cot_aokvqa-cot-stage0/epoch-8' into the container."
        ),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

