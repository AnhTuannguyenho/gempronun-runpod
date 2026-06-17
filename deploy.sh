#!/usr/bin/env bash
# Build + push image Gempronun cho RunPod. Chạy trên máy/VPS CÓ Docker.
#   IMAGE=docker.io/<user>/gempronun-runpod:1 ./deploy.sh
# Đăng nhập registry trước:  docker login   (Docker Hub) hoặc  docker login ghcr.io
set -euo pipefail

IMAGE="${IMAGE:?Set IMAGE=docker.io/<user>/gempronun-runpod:1}"
ASR_MODEL="${ASR_MODEL:-medium.en}"
PLATFORM="${PLATFORM:-linux/amd64}"   # RunPod chạy amd64; build trên Mac ARM phải ép cờ này

cd "$(dirname "$0")"

echo "==> build $IMAGE (model=$ASR_MODEL, platform=$PLATFORM)"
# buildx để ép kiến trúc amd64 (RunPod GPU là x86_64)
docker buildx build \
  --platform "$PLATFORM" \
  --build-arg "ASR_MODEL=$ASR_MODEL" \
  -t "$IMAGE" \
  --push \
  .

echo "==> done. Image pushed: $IMAGE"
echo "    Gắn image này vào endpoint RunPod (console -> endpoint -> Edit -> Container Image),"
echo "    hoặc dùng deploy_runpod_api.sh để cập nhật qua API."
