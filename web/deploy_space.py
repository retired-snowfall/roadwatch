"""Publish the website + live demo to a Hugging Face Docker Space.

Run by .github/workflows/deploy-space.yml on every push (or by hand). Needs one repository
secret, HF_TOKEN: a Hugging Face access token with write permission. The Space is created
on first run as <hf-user>/roadwatch (override with the HF_SPACE variable) and serves the
site at https://<hf-user>-roadwatch.hf.space.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "web", "weights", "solution.py", "evaluate.py", "requirements.txt", "LICENSE"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
SPACE_README = """---
title: RoadWatch
emoji: 🚦
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

RoadWatch: traffic-event detection and accident-risk estimation for a fixed road camera
(WIUT Hackathon 2026, computer-vision track). Source code: {repo}
"""


def main() -> int:
    token = (os.environ.get("HF_TOKEN") or "").strip().strip('"').strip("'")
    if not token:
        print("HF_TOKEN is not set: nothing to deploy (add it under Settings -> Secrets and variables -> Actions).")
        return 0
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        user = api.whoami()["name"]
    except Exception as exc:  # 401: the secret is not a valid token
        print(f"Hugging Face rejected HF_TOKEN ({exc.__class__.__name__}). The secret must be the token value itself "
              f"(starts with 'hf_', {len(token)} characters here, starts with {token[:3]!r}), created with the Write role.")
        return 1
    space = os.environ.get("HF_SPACE") or f"{user}/roadwatch"
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)

    stage = Path(tempfile.mkdtemp())
    for name in INCLUDE:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, stage / name, ignore=IGNORE)
        elif src.exists():
            shutil.copy2(src, stage / name)
    shutil.copy2(ROOT / "web" / "Dockerfile", stage / "Dockerfile")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    (stage / "README.md").write_text(SPACE_README.format(repo=f"https://github.com/{repo}" if repo else "see README"))

    sha = os.environ.get("GITHUB_SHA", "local")[:7]
    api.upload_folder(folder_path=str(stage), repo_id=space, repo_type="space", delete_patterns=["**"],
                      commit_message=f"deploy {sha}")
    host = space.replace("/", "-").replace("_", "-").lower()
    print(f"deployed to https://huggingface.co/spaces/{space}  (site: https://{host}.hf.space)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
