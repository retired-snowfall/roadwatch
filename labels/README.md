# Dev labels

`team_labels.txt` is the team's labelling of the four sample videos in plain text ("m:ss - m:ss class" under each
video name); `python tools/labels_from_text.py labels/team_labels.txt --videos samples/1080p` turns it into
`dev_labels.json`, merging overlapping events of one class as the official format requires.

A team member visited the junction afterwards: **U-turns are allowed there**, so the U-turns first labelled as
illegal turns were removed, except one (C3905, 1:34): a car that pushed through pedestrians on a crossing
and then drove onto the pavement, relabelled as failure_to_yield (driving on the pavement has no class).
There are no solid lines on the far carriageway (the next junction is about a kilometre away).

`dev_labels.json` holds our own annotations of the organisers' sample videos, in the official ground-truth format:

```json
{"sample_001.mp4": {"duration": 600.0, "fps": 25.0, "events": [[12.0, 19.0, "accident"]]}}
```

## How to label

1. Open the site's annotator (`annotate.html` on the hosted site, or locally: `uvicorn web.app:app --port 7860`,
   then http://localhost:7860/annotate.html).
2. Click one of the **sample videos hosted on the site** (browser-friendly copies: the camera's 4K 10-bit
   4:2:2 originals do not play in browsers). Our pipeline's events load as faded "pred" bars;
   **Use as starting labels** copies them in as a draft. Then correct it: delete false events, add missed ones,
   fix each start and end. Pick a class (keys 1–9, 0, Q, W, R, T), press **S** at the start and **E** at the end.
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
