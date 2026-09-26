#!/usr/bin/env bash
# One-command deployment of the website + live demo on a fresh Ubuntu 22.04/24.04 server
# (any cloud: 4 vCPU / 8 GB RAM / 40 GB disk recommended, no GPU, inbound TCP 22, 80, 443 open).
#
#   curl -fsSL https://raw.githubusercontent.com/retired-snowfall/some_shi/claude/zen-tesla-kuq5es/web/deploy_vm.sh | sudo bash
#
# It installs Docker, builds web/Dockerfile from the repository and serves it with automatic HTTPS
# (Caddy + Let's Encrypt) at https://<public-ip>.sslip.io, or at DOMAIN if you have one pointing
# to the server:  ... | sudo DOMAIN=roadwatch.example.uz bash
# Re-run the same command to update to the latest commit.
set -euo pipefail

REPO=${REPO:-https://github.com/retired-snowfall/some_shi.git}
BRANCH=${BRANCH:-claude/zen-tesla-kuq5es}
DIR=${DIR:-/opt/roadwatch}

echo "==> Docker and git"
if ! command -v docker >/dev/null; then
  apt-get update -y
  apt-get install -y docker.io git curl
  systemctl enable --now docker
fi
command -v git >/dev/null || apt-get install -y git

echo "==> Source ($BRANCH)"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch --depth 1 origin "$BRANCH"
  git -C "$DIR" checkout -q -B "$BRANCH" FETCH_HEAD
else
  git clone --depth 1 -b "$BRANCH" "$REPO" "$DIR"
fi

echo "==> Build the image (first time ~10 minutes)"
docker build -f "$DIR/web/Dockerfile" -t roadwatch-web "$DIR"

echo "==> Run"
docker rm -f roadwatch >/dev/null 2>&1 || true
docker run -d --name roadwatch --restart unless-stopped -p 127.0.0.1:7860:7860 \
  -e DEMO_MAX_SEC="${DEMO_MAX_SEC:-180}" -e DEMO_MAX_MB="${DEMO_MAX_MB:-200}" roadwatch-web

IP=$(curl -fsS https://api.ipify.org || hostname -I | awk '{print $1}')
HOST=${DOMAIN:-$IP.sslip.io}
echo "==> HTTPS for $HOST"
docker rm -f caddy >/dev/null 2>&1 || true
docker run -d --name caddy --restart unless-stopped --network host -v caddy_data:/data caddy:2 \
  caddy reverse-proxy --from "$HOST" --to 127.0.0.1:7860

echo "==> Waiting for the app"
for _ in $(seq 1 60); do
  curl -fsS http://127.0.0.1:7860/api/health >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS http://127.0.0.1:7860/api/health && echo
echo
echo "Website and live demo: https://$HOST"
echo "(if HTTPS is not ready within a minute, check that ports 80 and 443 are open in the cloud firewall)"
