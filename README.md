# roadwatch — traffic events and accident anticipation for a fixed road camera

WIUT Hackathon 2026, Computer Vision track. Given a video from one fixed CCTV view of a road, roadwatch

* **Part A** reports every traffic event as `[start_sec, end_sec, label]` for the 14 official classes;
* **Part B** returns, frame by frame and using only past frames, the probability that an accident starts within 5 s.

A COCO-pretrained YOLO11 detector and a ByteTrack-style tracker turn the video into trajectories. The camera is
re-aimed between recordings, so each video is registered to a reference view of the junction (SIFT + RANSAC
homography on its background). A model of the junction, learned without labels from the sample videos, says
where the carriageway is, which way each lane flows and where traffic queues (stop lines); the crosswalks are traced
once in the reference view. One rule module per family of classes reads the trajectories against it.

Website (team, approach, EDA, results, live demo, report): see `web/` and [web/DEPLOY.md](web/DEPLOY.md).

## Run it

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

* Python 3.10+. `torch==2.8.0` from PyPI ships CUDA 12.8 kernels (T4 / sm_75 supported). If no usable GPU is
  found the pipeline switches to its CPU profile automatically (smaller detector, lower analysis rate).
* **Weights are in the repository** (`weights/yolo11m.pt` 40 MB, `weights/yolo11n.pt` 5.6 MB, and the scene model
  `weights/scene.json` produced by `tools/calibrate.py` from the sample videos).
  If they are missing (e.g. a clone without large files), run `bash weights/download.sh` once with internet; it
  fetches them from the Ultralytics GitHub release and checks SHA-256 sums. Nothing is downloaded at run time
  (`YOLO_OFFLINE=true`).
* Docker alternative: `docker build -t roadwatch .` then
  `docker run --gpus all -v /data/test:/data/test -v $PWD/out:/out roadwatch`.
* `run_submission.py` and `evaluate.py` are the organisers' files, unchanged (copies are kept in
  `scripts/wiut_cv_scripts/`).

## Approach

```
frames ──► YOLO11 detector ──► ByteTrack-style tracker ──► trajectories (stitched, smoothed, sizes/s)
 (8 fps GPU / 5 fps CPU)       (learned, COCO)              (classical)                   │
                                                              register the view to the reference (homography)
                                                                                          ▼
  1 fps pixel side channels ─────────────────────────►  rule modules per class  ◄── scene model of the camera
  (background, static objects, fire, smoke, lights)          (hand-written)         (learned from the samples
                                                                   │                  without labels + drawn zones)
                                                                   ▼
                                     merge < 1 s gaps, drop < 0.8 s, clip ──► [[start, end, label], ...]

Part B (causal):  frame ──► detector (6 fps) ──► online tracker ──► TTC, braking, wrong-way, impact cues
                  ──► 1 − ∏(1 − cue) with fast attack / held release ──► risk in [0, 1]
```

| Component | Learned or rule-based | Where |
|---|---|---|
| Video decoding | classical: PyAV (FFmpeg) skips non-reference frames and scales to a fixed 1920-px working width in a background thread; all geometry is in working pixels, so 4K and 1080p copies of the camera agree | `video.py` |
| Road-user and traffic-light detection | learned: YOLO11-m (GPU, 1280 px) / YOLO11-n (CPU, 640 px), COCO-pretrained, not fine-tuned | `src/roadwatch/detector.py` |
| Tracking | classical: Kalman filter + two-stage association (ByteTrack), class-group gating, centre-distance fallback | `tracker.py` |
| Trajectories | classical: fragment stitching, rider suppression, local-linear smoothing, speeds in object sizes/s | `tracks.py` |
| View registration | classical: SIFT on CLAHE-equalised background, RANSAC homography to the reference view (rejects other cameras); rules run in reference coordinates, overlays are mapped back | `register.py` |
| Scene model | learned from data without labels: carriageway, per-cell lane direction, streams, stop zones, queue heads → stop lines, movement statistics; hand-drawn zones (`weights/zones.json`) override | `scene.py` |
| Signal state | rule-based: queue heads waiting at the line; traffic-light colour (HSV) when the heads are visible | `signals.py` |
| Event classes | rule-based, one module per family | `events/*.py` |
| Obstacles, fire, smoke | classical pixel analysis at 1 fps | `appearance.py` |
| Part B | rule-based cues on a causal tracker | `risk.py` |

Per-class rules and their start/end conventions are documented on the website's Approach page and in the module
docstrings. Every rule prefers precision: a class predicted but absent from the test set adds a zero to the macro F1.

### Findings on the real junction

The first run on the organisers' camera reported 42 events in two minutes of ordinary traffic. Reviewing every
class on contact sheets (three frames per event, tracks drawn in) showed the causes: identity switches between
queued cars, boxes that overlap in perspective without touching, turns read as swerves or wrong-way driving,
coarse crossing areas, and a camera re-aimed between recordings. The fixes (motion-scaled tracker gate, track-quality
checks, view registration, traced crosswalks, stop lines fitted along the waiting cars, stricter signal evidence)
brought the four samples from 93 to 14 detections; the website's report page has the per-class table and figures.
Two sources that fired only on artefacts here are kept but off by default: statistically rare movements as illegal
turns (`EventCfg.it_rare_movements`) and background-difference obstacles (`EventCfg.ob_static_blobs`).

### Time budget

Limit: 3 × video duration for Part A + Part B together; `run_submission.py` scores a video that runs over as
empty (events *and* risk). The organisers' camera writes 4K 10-bit 4:2:2 H.264 at ~140 Mbit/s, and decoding it
on the CPU is the main cost (NVDEC does not decode 4:2:2 H.264):

* Part A decodes with PyAV, skipping non-reference frames (2 of every 3 in this camera's GOP) and scaling to a
  1920-px working width in a background thread, so decoding overlaps detection; it analyses 8 frames/s in
  batches of 8 (FP16 on GPU).
* Part B receives every full-size frame from the harness (its OpenCV decode is the largest single cost and
  outside our control) and detects 6 frames/s at ≤ 960 px (CPU: 5 and 3 frames/s).
* Guards: Part A lowers its rate if it runs slower than 1.6× real time; Part B paces itself against the same
  per-video clock (`src/roadwatch/budget.py`) and lowers its rate if its projected finish nears 85 % of the budget.

Measured with the unchanged harness on the 4K original of C3905 (127.6 s), **4 CPU cores and no GPU**:
Part A 106 s (0.83×), Part B 160 s (1.26×), total 267 s = **2.09×** real time. The target machine has 8 cores
and a T4, so both decoding and detection are faster there.

### Determinism

Seeds are fixed (`SEED = 1234` in `src/roadwatch/config.py`; Python, NumPy, torch), cuDNN runs in deterministic
mode, frames are selected by timestamp (not wall-clock), and the tracker and rules are deterministic. The only
wall-clock-dependent behaviour is the pair of slow-machine guards above, which do not trigger at the configured
rates on the target hardware.

## Reproduce the calibration, the website data and `predictions_samples.json`

Put the organisers' sample videos in `samples/` (git-ignored; they are several GB), then:

```bash
python tools/calibrate.py --videos samples            # -> weights/scene.json, weights/scene_background.jpg
python run_submission.py --videos samples --out predictions_samples.json --team roadwatch
python tools/build_site.py --videos samples           # -> web/static/data/ (results, EDA, annotated videos)
python tools/evaluate_dev.py --videos samples --labels labels/dev_labels.json \
    --report web/static/data/dev_scores.json          # our labels, official metric, class confusion
```

`tools/calibrate.py` uses the first video (or `--reference NAME`) as the reference view and registers the others to
it; `weights/scene_background.jpg` is that view's background. Road-layout zones that cannot be learned reliably
(crosswalks, solid lines, prohibited turns, where U-turns are allowed) are drawn once on it in the website's scene
editor (`web/static/scene.html`) and saved as `weights/zones.json`; the junction's three crosswalks are traced there.
The organisers' files are 4K 10-bit 4:2:2 (≈140 Mbit/s); for development we used 1080p H.264 copies
(`ffmpeg -i IN -an -vf scale=1920:1080 -c:v libx264 -crf 18 OUT`), which give the same geometry because the
pipeline works at a 1920-px working width either way.
Dev labels are made with the in-browser annotator (`web/static/annotate.html`), which exports the official
ground-truth format. Perception is cached in `.cache/perception` when `ROADWATCH_CACHE` is set (the tools set it),
so re-scoring after a rule change takes seconds.

## Datasets and licences

| Item | Use | Licence |
|---|---|---|
| YOLO11 weights and `ultralytics` (Ultralytics) | detector, pretrained on COCO, not fine-tuned | AGPL-3.0 |
| COCO 2017 | pre-training of the detector (by Ultralytics) | CC BY 4.0 (annotations) |
| Organisers' sample videos | scene calibration, EDA, our dev labels | competition use |
| Intel IoT sample videos; `ahmetozlu/tensorflow_object_counting_api` sample clips | local smoke tests only, not in the repository | CC BY 4.0; MIT |

No other dataset is used; no model is trained on accident data. Reused code: the tracker follows ByteTrack
(Zhang et al., 2022, MIT licence), re-implemented for irregular time steps. OpenCV (Apache-2.0), NumPy, SciPy
(BSD), FastAPI (MIT). Ultralytics is AGPL-3.0, so redistributing this repository together with it falls under
AGPL-3.0 terms.

## Repository layout

```
solution.py              interface for the harness (thin adapter over src/roadwatch)
run_submission.py        organisers' harness, unchanged
evaluate.py              organisers' metric, unchanged
requirements.txt         inference dependencies (pinned)
Dockerfile               optional container route for the submission
weights/                 detector weights, scene.json (learned), zones.json (drawn), download.sh
src/roadwatch/           the pipeline
  video.py detector.py tracker.py tracks.py      perception
  scene.py signals.py appearance.py              scene model and side channels
  events/ collision.py motion.py stationary.py pedestrian.py lines.py hazards.py base.py
  pipeline.py            Part A orchestration          risk.py    Part B
  eda.py render.py       website statistics, annotated videos
tools/                   calibrate.py, build_site.py, evaluate_dev.py
tests/                   rule tests on synthetic trajectories, tracker/scene/interface tests
web/                     FastAPI demo backend (app.py, jobs.py), static site (static/), Dockerfile, DEPLOY.md
```

## Development

```bash
pip install -r requirements.txt pytest
python -m pytest tests -q
uvicorn web.app:app --port 7860      # website + live demo at http://localhost:7860
```

## Team

| Member | Role | Contributions |
|---|---|---|
| Member 1 (to fill) | Perception & tracking | detector and tracker integration, time budget |
| Member 2 (to fill) | Event rules & evaluation | rules for the 14 classes, dev labels, tuning |
| Member 3 (to fill) | Website, EDA & report | website and live demo, EDA, report |
