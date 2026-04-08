#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VARIANT="${1:-docbert}"

if [[ "$VARIANT" != "docbert" && "$VARIANT" != "docbert-cuda" ]]; then
  echo "Usage: $0 [docbert|docbert-cuda]"
  exit 1
fi

echo "==> Testing PKGBUILD for $VARIANT"

docker build -t "docbert-pkgbuild-test:${VARIANT}" -f - "$REPO_ROOT" <<'DOCKERFILE'
FROM archlinux:base-devel

RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm git rustup

RUN useradd -m builder && \
    echo 'builder ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

USER builder
RUN rustup default stable
WORKDIR /home/builder
DOCKERFILE

docker run --rm \
  -v "${SCRIPT_DIR}/${VARIANT}/PKGBUILD:/home/builder/PKGBUILD.src:ro" \
  "docbert-pkgbuild-test:${VARIANT}" \
  bash -c 'mkdir ~/build && cp ~/PKGBUILD.src ~/build/PKGBUILD && cd ~/build && makepkg -si --noconfirm && docbert --help'

echo "==> Build succeeded for $VARIANT"
