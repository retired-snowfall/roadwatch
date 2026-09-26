"""Publish the website to a Hugging Face Space.

Run by .github/workflows/deploy-space.yml on every push (or by hand). Needs one repository
secret, HF_TOKEN: a Hugging Face access token with write permission. The Space is created
on first run as <hf-user>/roadwatch (override with the HF_SPACE variable) and serves the
site at https://<hf-user>-roadwatch.hf.space.

Two modes (repository variable HF_SPACE_SDK; the workflow defaults to docker):
  docker   the full FastAPI app (web/Dockerfile): site and live demo in one Space. Needs Hugging
           Face PRO on the account; runs on the free CPU Basic hardware.
  static   the static site (all pages, results, videos); free on every account. The live demo
           page calls the backend named by the ROADWATCH_API variable (a server running
           web/Dockerfile, e.g. set up with web/deploy_vm.sh).
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "web", "weights", "solution.py", "evaluate.py", "requirements.txt", "LICENSE"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
HEADER = {"docker": "sdk: docker\napp_port: 7860", "static": "sdk: static"}
SPACE_README = """---
title: RoadWatch
emoji: 🚦
colorFrom: blue
colorTo: gray
{sdk}
pinned: false
---

RoadWatch: traffic-event detection and accident-risk estimation for a fixed road camera
(WIUT Hackathon 2026, computer-vision track). Source code: {repo}
"""


def stage_docker(stage: Path) -> None:
    for name in INCLUDE:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, stage / name, ignore=IGNORE)
        elif src.exists():
            shutil.copy2(src, stage / name)
    shutil.copy2(ROOT / "web" / "Dockerfile", stage / "Dockerfile")


def stage_static(stage: Path, api: str) -> None:
    shutil.copytree(ROOT / "web" / "static", stage, ignore=IGNORE, dirs_exist_ok=True)
    if api:  # point the live demo at the backend server
        demo = stage / "demo.html"
        html = demo.read_text()
        demo.write_text(re.sub(r'<meta name="roadwatch-api" content="[^"]*">',
                               f'<meta name="roadwatch-api" content="{api.rstrip("/")}">', html))


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
    sdk = (os.environ.get("HF_SPACE_SDK") or "static").strip().lower()
    if sdk not in HEADER:
        print(f"HF_SPACE_SDK must be 'static' or 'docker', not {sdk!r}")
        return 1
    space = (os.environ.get("HF_SPACE") or "").strip() or f"{user}/roadwatch"
    api.create_repo(space, repo_type="space", space_sdk=sdk, exist_ok=True)

    stage = Path(tempfile.mkdtemp())
    if sdk == "docker":
        stage_docker(stage)
    else:
        stage_static(stage, (os.environ.get("ROADWATCH_API") or "").strip())
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    (stage / "README.md").write_text(SPACE_README.format(sdk=HEADER[sdk],
                                                         repo=f"https://github.com/{repo}" if repo else "see README"))

    sha = os.environ.get("GITHUB_SHA", "local")[:7]
    api.upload_folder(folder_path=str(stage), repo_id=space, repo_type="space", delete_patterns=["**"],
                      commit_message=f"deploy {sha} ({sdk})")
    host = space.replace("/", "-").replace("_", "-").lower()
    site = f"https://{host}.static.hf.space" if sdk == "static" else f"https://{host}.hf.space"
    print(f"deployed to https://huggingface.co/spaces/{space}  (site: {site})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
