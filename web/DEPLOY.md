# Deploying the website and live demo

The site is one FastAPI app: static pages from `web/static/` plus the demo API (`web/app.py`). It runs on CPU;
a 2-minute 1080p upload takes roughly 2–4 minutes on 2 vCPUs. It must stay online through the judging period,
so pick a host that does not sleep, or keep one visit per day.

## 0. Build the site data first

```bash
python tools/calibrate.py --videos samples
python run_submission.py --videos samples --out predictions_samples.json --team roadwatch
python tools/build_site.py --videos samples
python tools/evaluate_dev.py --videos samples --labels labels/dev_labels.json --report web/static/data/dev_scores.json
```

`web/static/data/**/*.mp4` (previews and annotated renders) are git-ignored because of their size; the host needs
them, so upload them together with the image (Hugging Face: via git-lfs, see below).

## Option A — Hugging Face Spaces via GitHub Actions (recommended)

`.github/workflows/deploy-space.yml` runs `web/deploy_space.py` on every push: it creates the Space
`<hf-user>/roadwatch` (Docker SDK, free CPU) on first run and uploads the site, the demo backend and the weights.
One-time setup:

1. Create a Hugging Face account; under *Settings → Access Tokens* create a token with **write** access.
2. In this GitHub repository: *Settings → Secrets and variables → Actions → New repository secret*,
   name `HF_TOKEN`, value the token. (Optional: a variable `HF_SPACE` to choose another Space name.)
3. *Actions → Deploy website → Run workflow* (or push). The site appears at `https://<hf-user>-roadwatch.hf.space`
   after the image builds (~10 minutes the first time).

## Option A′ — Hugging Face Spaces by hand

1. Create a Space: SDK **Docker**, hardware **CPU basic**.
2. Clone it, copy this repository into it, and make the web Dockerfile the Space's root Dockerfile:
   ```bash
   git clone https://huggingface.co/spaces/<user>/roadwatch && cd roadwatch
   rsync -a --exclude .git --exclude samples --exclude .cache ../some_shi/ .
   cp web/Dockerfile Dockerfile
   printf -- '---\ntitle: roadwatch\nsdk: docker\napp_port: 7860\n---\n' | cat - README.md > README.tmp && mv README.tmp README.md
   git lfs install && git lfs track "*.mp4" "*.pt" && git add .gitattributes
   git add -A && git commit -m "deploy" && git push
   ```
3. The Space builds the image and serves the site at `https://<user>-roadwatch.hf.space`.

## Option B — Render / Railway / any Docker host

Point the service at this repository with Dockerfile path `web/Dockerfile` and build context `.`; the container
listens on `$PORT` (default 7860). Add the generated `web/static/data` media to the image or a mounted disk.

## Option C — static site on GitHub Pages + demo elsewhere

All pages except the demo are static. Publish `web/static/` on GitHub Pages and set the demo backend URL in
`web/static/demo.html`: `<meta name="roadwatch-api" content="https://<your-demo-host>">`. The API sends CORS headers.

## Limits (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `DEMO_MAX_MB` | 200 | largest upload |
| `DEMO_MAX_SEC` | 180 | longest video |
| `DEMO_JOBS_DIR` | `.cache/jobs` | uploads, previews, results (deleted after 3 hours) |
| `ROADWATCH_DEVICE` | auto | force `cpu` or `gpu` |
