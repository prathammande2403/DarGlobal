#!/bin/sh
set -eu

if [ ! -f "/app/data/chroma_updated/chroma.sqlite3" ]; then
    echo "The image does not contain a Chroma index. Rebuild it with docker compose build."
    exit 1
fi

exec "$@"
