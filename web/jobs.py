"""Background job queue for the live demo: one worker, staged progress, automatic cleanup."""
from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from roadwatch.pipeline import analyze
from roadwatch.render import transcode_preview
from roadwatch.risk import risk_curve

log = logging.getLogger("roadwatch.web")

# share of the total work per stage (CPU profile), used for the progress bar and ETA
STAGES = [("preview", 0.05), ("detect", 0.62), ("tracks", 0.02), ("rules", 0.03), ("risk", 0.28)]


@dataclass
class Job:
    id: str
    video: Path
    name: str
    duration: float
    status: str = "queued"          # queued | running | done | error
    stage: str = "queued"
    stage_progress: float = 0.0
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    @property
    def dir(self) -> Path:
        return self.video.parent

    @property
    def progress(self) -> float:
        if self.status == "done":
            return 1.0
        done = 0.0
        for name, w in STAGES:
            if name == self.stage:
                return min(0.99, done + w * self.stage_progress)
            done += w
        return 0.0

    def public(self, position: int | None) -> dict:
        eta = None
        if self.status == "running" and self.started and self.progress > 0.03:
            elapsed = time.time() - self.started
            eta = round(elapsed / self.progress - elapsed, 1)
        return {"id": self.id, "name": self.name, "duration": round(self.duration, 2), "status": self.status,
                "stage": self.stage, "progress": round(self.progress, 3), "eta_sec": eta, "queue_position": position,
                "error": self.error,
                "elapsed_sec": round((self.finished or time.time()) - self.started, 1) if self.started else None}


class JobQueue:
    def __init__(self, root: Path, keep_sec: float = 3 * 3600):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_sec = keep_sec
        self.jobs: dict[str, Job] = {}
        self.pending: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True, name="demo-worker").start()

    def new_dir(self) -> tuple[str, Path]:
        jid = uuid.uuid4().hex[:12]
        d = self.root / jid
        d.mkdir(parents=True)
        return jid, d

    def submit(self, job: Job) -> None:
        with self.lock:
            self.jobs[job.id] = job
        self.pending.put(job.id)

    def position(self, jid: str) -> int | None:
        with self.lock:
            waiting = [j for j in self.jobs.values() if j.status == "queued"]
        waiting.sort(key=lambda j: j.created)
        ids = [j.id for j in waiting]
        return ids.index(jid) + 1 if jid in ids else None

    def get(self, jid: str) -> Job | None:
        return self.jobs.get(jid)

    # ------------------------------------------------------------------ worker
    def _worker(self) -> None:
        while True:
            jid = self.pending.get()
            job = self.jobs.get(jid)
            if job is None:
                continue
            job.status, job.started = "running", time.time()
            try:
                self._run(job)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - reported to the visitor
                log.error("job %s failed\n%s", jid, traceback.format_exc())
                job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
            job.finished = time.time()
            self._cleanup()

    def _run(self, job: Job) -> None:
        def progress(stage: str, frac: float) -> None:
            if stage == "done":
                return
            job.stage, job.stage_progress = stage, frac

        progress("preview", 0.0)
        transcode_preview(job.video, job.dir / "preview.mp4", max_width=960)
        an = analyze(str(job.video), progress=progress)
        result = an.to_json()
        result["video"] = job.name
        progress("risk", 0.0)
        curve = risk_curve(str(job.video), progress=progress)
        step = max(1, len(curve) // 3000)
        result["risk"] = curve[::step]
        (job.dir / "result.json").write_text(json.dumps(result))
        job.video.unlink(missing_ok=True)          # keep only the preview and the result

    def _cleanup(self) -> None:
        now = time.time()
        with self.lock:
            old = [j for j in self.jobs.values() if j.finished and now - j.finished > self.keep_sec]
            for j in old:
                shutil.rmtree(j.dir, ignore_errors=True)
                del self.jobs[j.id]
