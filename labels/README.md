# Dev labels

`dev_labels.json` holds our own annotations of the organisers' sample videos, in the official ground-truth format:

```json
{"sample_001.mp4": {"duration": 600.0, "fps": 25.0, "events": [[12.0, 19.0, "accident"]]}}
```

## How to label

1. Serve the site (`uvicorn web.app:app --port 7860`) and open http://localhost:7860/annotate.html
   (the video never leaves your machine).
2. Open a sample video, pick a class (keys 1–9, 0, Q, W, R, T), press **S** at the start and **E** at the end.
   Space plays/pauses, ←/→ step one frame, Shift+←/→ one second.
3. Follow the start/end conventions in the page's "Labelling conventions" box — they are the organisers'.
   Two simultaneous events of the same class are one segment; an event that runs past the end of the video
   ends at the video duration.
4. Export and save as `labels/dev_labels.json`. Split the videos between two people, then swap and review:
   boundaries matter at IoU 0.7, so agree on them.

## Score

```bash
python tools/evaluate_dev.py --videos samples --labels labels/dev_labels.json --report web/static/data/dev_scores.json
```
