#!/usr/bin/env bash
# Fetch detector weights (only needed if weights/*.pt are missing, e.g. a shallow or LFS-less clone).
# Run once, with internet, before the offline evaluation.
set -euo pipefail
cd "$(dirname "$0")"
BASE=https://github.com/ultralytics/assets/releases/download/v8.3.0
for m in yolo11m yolo11n; do
  [ -s "$m.pt" ] || curl -fL --retry 3 -o "$m.pt" "$BASE/$m.pt"
done
sha256sum -c SHA256SUMS
