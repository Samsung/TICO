#!/bin/bash

set -e

echo "[download-release-asset] Start"

# Input parameters
OWNER=$1
REPO=$2
TAG=$3
FILENAME=$4

echo "[download-release-asset] Target filename: $FILENAME"

API_URL="https://api.github.com/repos/${OWNER}/${REPO}/releases/tags/${TAG}"
DOWNLOAD_URL=$(curl -s $API_URL | jq -r ".assets[] | select(.name == \"$FILENAME\") | .browser_download_url")

if [ -z "$DOWNLOAD_URL" ]; then
  echo "[download-release-asset] ERROR: file not found in release assets: $FILENAME"
  exit 1
fi

echo "[download-release-asset] Downloading from: $DOWNLOAD_URL"
curl -L -o "$FILENAME" "$DOWNLOAD_URL"
echo "filename=$FILENAME" >> "$GITHUB_OUTPUT"

echo "[download-release-asset] End"
