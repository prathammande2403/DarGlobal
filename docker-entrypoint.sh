#!/bin/sh
set -eu

INDEX_DIR="/app/data/chroma_updated"
READY_MARKER="${INDEX_DIR}/.initialized"

if [ ! -f "$READY_MARKER" ]; then
    echo "Building the Chroma index from data/raw/darglobal.json (first run downloads the FastEmbed model)."
    python -m rag.ingest
    touch "$READY_MARKER"
fi

exec "$@"
