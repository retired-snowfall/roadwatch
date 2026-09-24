"""Team website + live demo backend.

    uvicorn web.app:app --host 0.0.0.0 --port 7860

Serves the static site from web/static and a small API:
    POST /api/jobs            upload a video (multipart field "file") -> {"id": ...}
    GET  /api/jobs/{id}       status, stage, progress, ETA
    GET  /api/jobs/{id}/result   events, evidence, tracks, scene layers, risk curve
    GET  /api/jobs/{id}/video    browser-safe H.264 preview of the upload
    GET  /api/health
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from roadwatch.detector import device_kind, get_detector  # noqa: E402
from roadwatch.video import probe  # noqa: E402

from .jobs import Job, JobQueue  # noqa: E402

MAX_MB = float(os.environ.get("DEMO_MAX_MB", 200))
MAX_SEC = float(os.environ.get("DEMO_MAX_SEC", 180))
ALLOWED = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
STATIC = Path(__file__).resolve().parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app = FastAPI(title="roadwatch demo", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
jobs = JobQueue(Path(os.environ.get("DEMO_JOBS_DIR", ROOT / ".cache" / "jobs")))


@app.on_event("startup")
def warm_up() -> None:
    get_detector()          # load weights once, before the first visitor waits for it


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "device": device_kind(), "queued": jobs.pending.qsize(),
            "limits": {"max_mb": MAX_MB, "max_sec": MAX_SEC, "formats": sorted(ALLOWED)}}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, f"unsupported file type {suffix}; use one of {', '.join(sorted(ALLOWED))}")
    jid, d = jobs.new_dir()
    dest = d / f"upload{suffix}"
    size = 0
    with open(dest, "wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_MB * 1e6:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"file larger than {MAX_MB:.0f} MB")
            out.write(chunk)
    try:
        info = probe(str(dest))
    except IOError:
        raise HTTPException(422, "could not decode this video")
    if info.duration <= 0 or info.n_frames <= 0:
        raise HTTPException(422, "video has no frames")
    if info.duration > MAX_SEC + 1:
        raise HTTPException(413, f"video is {info.duration:.0f} s long; the demo accepts up to {MAX_SEC:.0f} s")
    jobs.submit(Job(jid, dest, Path(file.filename or "upload.mp4").name, info.duration))
    return {"id": jid}


def _job(jid: str) -> Job:
    job = jobs.get(jid)
    if job is None:
        raise HTTPException(404, "unknown or expired job")
    return job


@app.get("/api/jobs/{jid}")
def job_status(jid: str) -> dict:
    return _job(jid).public(jobs.position(jid))


@app.get("/api/jobs/{jid}/result")
def job_result(jid: str) -> FileResponse:
    job = _job(jid)
    path = job.dir / "result.json"
    if job.status != "done" or not path.exists():
        raise HTTPException(409, f"job is {job.status}")
    return FileResponse(path, media_type="application/json")


@app.get("/api/jobs/{jid}/video")
def job_video(jid: str) -> FileResponse:
    path = _job(jid).dir / "preview.mp4"
    if not path.exists():
        raise HTTPException(409, "preview not ready")
    return FileResponse(path, media_type="video/mp4")


app.mount("/", StaticFiles(directory=STATIC, html=True), name="site")
