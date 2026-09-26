"""Upload a video to a running site's live demo and wait for the result (standard library only).

    python web/smoke_test.py https://<user>-roadwatch.hf.space web/static/data/samples/C3905/preview.mp4

Exits 0 when the job finishes with a result, 1 otherwise. Run by web/deploy_space.py after a Docker deploy.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path


def _request(url: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def upload(site: str, video: Path) -> str:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{video.name}\"\r\n"
            f"Content-Type: video/mp4\r\n\r\n").encode() + video.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return _request(f"{site}/api/jobs", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"},
                    timeout=600)["id"]


def run(site: str, video: Path, timeout: float = 25 * 60) -> int:
    site = site.rstrip("/")
    t0 = time.time()
    jid = upload(site, video)
    print(f"uploaded {video.name} ({video.stat().st_size / 1e6:.1f} MB) -> job {jid}")
    last = None
    while time.time() - t0 < timeout:
        st = _request(f"{site}/api/jobs/{jid}")
        line = f"{st.get('status')} {st.get('stage', '')} {int(st.get('progress', 0) * 10) * 10}%"
        if line != last:
            print(f"[{time.time() - t0:5.0f}s] {line}")
            last = line
        if st.get("status") == "done":
            res = _request(f"{site}/api/jobs/{jid}/result")
            risk = res.get("risk", [])
            print(f"result: {len(res.get('events', []))} events {[e[2] for e in res.get('events', [])]}, "
                  f"{len(risk)} risk points, max risk {max((r[1] for r in risk), default=0):.2f}, "
                  f"{time.time() - t0:.0f} s end to end")
            return 0
        if st.get("status") == "error":
            print("job failed:", st.get("error"))
            return 1
        time.sleep(10)
    print("timed out")
    return 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1], Path(sys.argv[2])))
